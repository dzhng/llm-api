import { defaults } from 'lodash';
import {
  Configuration,
  CreateChatCompletionRequest,
  OpenAIApi,
} from 'openai-edge';

import {
  CompletionDefaultRetries,
  CompletionDefaultTimeout,
  DefaultOpenAIModel,
  MinimumResponseTokens,
  RateLimitRetryIntervalMs,
} from '../config';
import type {
  ModelRequestOptions,
  ModelConfig,
  OpenAIConfig,
  ChatRequestMessage,
  ChatResponse,
} from '../types';
import { debug, parseUnsafeJson, sleep } from '../utils';

import { TokenError } from './errors';
import type { CompletionApi } from './interface';
import { getTikTokenTokensFromPrompt } from './tokenizer';

const RequestDefaults = {
  retries: CompletionDefaultRetries,
  retryInterval: RateLimitRetryIntervalMs,
  timeout: CompletionDefaultTimeout,
  minimumResponseTokens: MinimumResponseTokens,
};
const AzureQueryParams = { 'api-version': '2023-03-15-preview' };

const convertConfig = (
  config: Partial<ModelConfig>,
): Partial<CreateChatCompletionRequest> => ({
  model: config.model,
  temperature: config.temperature,
  top_p: config.topP,
  n: 1,
  stop: config.stop,
  presence_penalty: config.presencePenalty,
  frequency_penalty: config.frequencyPenalty,
  logit_bias: config.logitBias,
  user: config.user,
  stream: config.stream,
});

export class OpenAIChatApi implements CompletionApi {
  _client: OpenAIApi;
  _isAzure: boolean;
  _headers?: Record<string, string>;
  modelConfig: ModelConfig;

  constructor(config: OpenAIConfig, modelConfig?: ModelConfig) {
    this._isAzure = Boolean(config.azureEndpoint && config.azureDeployment);

    const configuration = new Configuration({
      ...config,
      basePath: this._isAzure
        ? `${config.azureEndpoint}${
            config.azureEndpoint?.at(-1) === '/' ? '' : '/'
          }openai/deployments/${config.azureDeployment}`
        : undefined,
    });

    this._headers = this._isAzure
      ? { 'api-key': String(config.apiKey) }
      : undefined;

    const azureFetch: typeof globalThis.fetch = (input, init) => {
      const customInput =
        typeof input === 'string'
          ? `${input}?${new URLSearchParams(AzureQueryParams)}`
          : input instanceof URL
          ? `${input.toString()}?${new URLSearchParams(AzureQueryParams)}`
          : input;
      return fetch(customInput, init);
    };

    this._client = new OpenAIApi(
      configuration,
      undefined,
      this._isAzure ? azureFetch : undefined,
    );

    this.modelConfig = modelConfig ?? {};
  }

  getTokensFromPrompt = getTikTokenTokensFromPrompt;

  // eslint-disable-next-line complexity
  async chatCompletion(
    initialMessages: ChatRequestMessage[],
    requestOptions = {} as Partial<ModelRequestOptions>,
  ): Promise<ChatResponse> {
    const finalRequestOptions = defaults(requestOptions, RequestDefaults);
    const messages: ChatRequestMessage[] = finalRequestOptions.systemMessage
      ? [
          {
            role: 'system',
            content:
              typeof finalRequestOptions.systemMessage === 'string'
                ? finalRequestOptions.systemMessage
                : finalRequestOptions.systemMessage(),
          },
          ...initialMessages,
        ]
      : initialMessages;

    debug.log(
      `🔼 completion requested: ${JSON.stringify(
        messages,
      )}, config: ${JSON.stringify(
        this.modelConfig,
      )}, options: ${JSON.stringify(finalRequestOptions)}`,
    );

    try {
      // check if we'll have enough tokens to meet the minimum response
      const maxPromptTokens = this.modelConfig.contextSize
        ? this.modelConfig.contextSize -
          finalRequestOptions.minimumResponseTokens
        : 1_000_000;

      const messageTokens = this.getTokensFromPrompt(
        messages.map((m) => m.content ?? ''),
        finalRequestOptions.functions,
      );
      if (messageTokens > maxPromptTokens) {
        throw new TokenError(
          'Prompt too big, not enough tokens to meet minimum response',
          messageTokens - maxPromptTokens,
        );
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        finalRequestOptions.timeout,
      );
      const completion = await this._client.createChatCompletion(
        {
          model: DefaultOpenAIModel,
          ...convertConfig(this.modelConfig),
          functions: finalRequestOptions.functions,
          function_call: finalRequestOptions.callFunction
            ? { name: finalRequestOptions.callFunction }
            : finalRequestOptions.functions
            ? 'auto'
            : undefined,
          messages,
        },
        {
          signal: controller.signal,
          headers: this._headers,
        },
      );
      clearTimeout(timeoutId);

      if (!completion.ok) {
        if (completion.status === 401) {
          debug.error(
            'Authorization error, did you set the OpenAI API key correctly?',
          );
          throw new Error('Authorization error');
        } else if (completion.status === 429 || completion.status >= 500) {
          debug.log(
            `Completion rate limited (${completion.status}), retrying... attempts left: ${finalRequestOptions.retries}`,
          );
          await sleep(finalRequestOptions.retryInterval);
          return this.chatCompletion(messages, {
            ...finalRequestOptions,
            retries: finalRequestOptions.retries - 1,
            // double the interval everytime we retry
            retryInterval: finalRequestOptions.retryInterval * 2,
          });
        }
      }

      let content: string | undefined;
      let functionCall: { name: string; arguments: string } | undefined;
      let usage: any;
      if (this.modelConfig.stream) {
        const reader = completion.body?.getReader();
        if (!reader) {
          throw new Error('Reader undefined');
        }

        const decoder = new TextDecoder('utf-8');
        while (true) {
          const { done, value } = await reader.read();
          const stringfied = decoder.decode(value).split('\n');

          for (const line of stringfied) {
            try {
              const cleaned = line.replace('data:', '').trim();
              if (cleaned.length === 0 || cleaned === '[DONE]') {
                continue;
              }

              const parsed = parseUnsafeJson(cleaned) as any;
              const text = parsed.choices[0].delta.content ?? '';

              debug.write(text);
              finalRequestOptions?.events?.emit('data', text);
              content += text;
            } catch (e) {
              debug.error('Error parsing content', e);
            }
          }

          if (done) {
            break;
          }
        }
        debug.write('\n[STREAM] response end\n');
      } else {
        const body = await completion.json();
        if (body.error || !('choices' in body)) {
          throw new Error(
            `Completion response error: ${JSON.stringify(body ?? {})}`,
          );
        }

        content = body.choices[0].message?.content;
        functionCall = body.choices[0].message?.function_call;
        usage = body.usage;

        debug.log('🔽 completion received', body.choices[0].message, usage);
      }

      if (content) {
        return {
          content,
          respond: (message: string | ChatRequestMessage, opt) =>
            this.chatCompletion(
              [
                ...messages,
                { role: 'assistant', content },
                typeof message === 'string'
                  ? { role: 'user', content: message }
                  : message,
              ],
              opt ?? requestOptions,
            ),
          usage: usage
            ? {
                totalTokens: usage.total_tokens,
                promptTokens: usage.prompt_tokens,
                completionTokens: usage.completion_tokens,
              }
            : undefined,
        };
      } else if (functionCall) {
        return {
          name: functionCall.name,
          arguments: parseUnsafeJson(functionCall.arguments),
          respond: (message: string | ChatRequestMessage, opt) =>
            this.chatCompletion(
              [
                ...messages,
                {
                  role: 'assistant',
                  content: '', // explicitly put empty string, or api will complain it's required property
                  function_call: functionCall,
                },
                typeof message === 'string'
                  ? { role: 'user', content: message }
                  : message,
              ],
              opt ?? requestOptions,
            ),
          usage: usage
            ? {
                totalTokens: usage.total_tokens,
                promptTokens: usage.prompt_tokens,
                completionTokens: usage.completion_tokens,
              }
            : undefined,
        };
      } else {
        throw new Error('Completion response malformed');
      }
    } catch (error: any) {
      // no more retries left
      if (!finalRequestOptions.retries) {
        debug.log('Completion failed, already retryed, failing completion');
        throw error;
      }

      if (
        error.code === 'ETIMEDOUT' ||
        error.code === 'ECONNABORTED' ||
        error.code === 'ECONNRESET'
      ) {
        debug.log(
          `Completion timed out (${error.code}), retrying... attempts left: ${finalRequestOptions.retries}`,
        );
        await sleep(finalRequestOptions.retryInterval);
        return this.chatCompletion(messages, {
          ...finalRequestOptions,
          retries: finalRequestOptions.retries - 1,
          // double the interval everytime we retry
          retryInterval: finalRequestOptions.retryInterval * 2,
        });
      }

      throw error;
    }
  }

  async textCompletion(
    prompt: string,
    requestOptions = {} as Partial<ModelRequestOptions>,
  ): Promise<ChatResponse> {
    const messages: ChatRequestMessage[] = [{ role: 'user', content: prompt }];
    return this.chatCompletion(messages, requestOptions);
  }
}
