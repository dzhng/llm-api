import { defaults } from 'lodash';
import {
  Configuration,
  CreateChatCompletionRequest,
  OpenAIApi,
} from 'openai-edge';

import {
  CompletionDefaultRetries,
  CompletionDefaultTimeout,
  DefaultAzureVersion,
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
  // NOTE: this is left without defaults by design - OpenAI's API will throw an error if max_token values is greater than model context size, which means it needs to be different for every model and cannot be set as a default. This fine since OpenAI won't put any limit on max_tokens if it's not set anyways (unlike Anthropic).
  // maximumResponseTokens: MaximumResponseTokens,
};

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

    const azureQueryParams = {
      'api-version': config.azureApiVersion ?? DefaultAzureVersion,
    };
    const azureFetch: typeof globalThis.fetch = (input, init) => {
      const customInput =
        typeof input === 'string'
          ? `${input}?${new URLSearchParams(azureQueryParams)}`
          : input instanceof URL
          ? `${input.toString()}?${new URLSearchParams(azureQueryParams)}`
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
    if (finalRequestOptions.responsePrefix) {
      console.warn('OpenAI models currently does not support responsePrefix');
    }

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
        : 100_000;

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

      // calculate max response tokens
      // note that for OpenAI models, it MUST be conditional on the contextSize being set, this is because OpenAI's API throws an error if maxTokens is above context size
      const maxTokens =
        this.modelConfig.contextSize &&
        finalRequestOptions.maximumResponseTokens
          ? Math.min(
              this.modelConfig.contextSize - maxPromptTokens,
              finalRequestOptions.maximumResponseTokens,
            )
          : undefined;
      if (
        finalRequestOptions.maximumResponseTokens &&
        !this.modelConfig.contextSize
      ) {
        console.warn(
          'maximumResponseTokens option ignored, please set contextSize in ModelConfig so the parameter can be calculated safely',
        );
      }

      const completion = await this._client.createChatCompletion(
        {
          model: DefaultOpenAIModel,
          ...convertConfig(this.modelConfig),
          max_tokens: maxTokens,
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

      let content = '';
      let functionCall: { name: string; arguments: string } | undefined;
      let usage: any;
      if (this.modelConfig.stream) {
        const reader = completion.body?.getReader();
        if (!reader) {
          throw new Error('Reader undefined');
        }

        const functionCallStreamParts: Partial<
          NonNullable<typeof functionCall>
        >[] = [];
        const decoder = new TextDecoder('utf-8');
        let lineBuffer = '';
        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            // finalize function call data from all parts from stream
            if (functionCallStreamParts.length > 0) {
              functionCall = functionCallStreamParts.reduce(
                (prev, part) => ({
                  name: (prev.name ?? '') + (part.name ?? ''),
                  arguments: (prev.arguments ?? '') + (part.arguments ?? ''),
                }),
                {},
              ) as { name: string; arguments: string };
            }
            break;
          }

          // make sure lineBuffer ends on a valid character before continuing (e.g. not in the middle of a packet)
          lineBuffer += decoder.decode(value);
          if (!lineBuffer.endsWith('\n')) {
            continue;
          }

          const stringified = lineBuffer.split('\n');
          lineBuffer = '';

          for (const line of stringified) {
            try {
              const cleaned = line.replace('data:', '').trim();
              if (cleaned.length === 0 || cleaned === '[DONE]') {
                continue;
              }

              const parsed = parseUnsafeJson(cleaned) as any;
              const text = parsed.choices[0].delta.content;
              const part = parsed.choices[0].delta.function_call as Partial<
                typeof functionCall
              >;

              const emitMessage: string = part
                ? part.name
                  ? `${part.name}: ${part.arguments}`
                  : part.arguments ?? ''
                : text ?? '';
              debug.write(emitMessage);
              finalRequestOptions?.events?.emit('data', emitMessage);

              if (text) {
                content += text;
              } else if (part) {
                functionCallStreamParts.push(part);
              }
            } catch (e) {
              debug.error('Error parsing content', e);
            }
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
        const receivedMessage: ChatRequestMessage = {
          role: 'assistant',
          content,
        };
        return {
          message: receivedMessage,
          content,
          respond: (message: string | ChatRequestMessage, opt) =>
            this.chatCompletion(
              [
                ...messages,
                receivedMessage,
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
        const receivedMessage: ChatRequestMessage = {
          role: 'assistant',
          content: '', // explicitly put empty string, or api will complain it's required property
          function_call: functionCall,
        };
        return {
          message: receivedMessage,
          name: functionCall.name,
          arguments: parseUnsafeJson(functionCall.arguments),
          respond: (message: string | ChatRequestMessage, opt) =>
            this.chatCompletion(
              [
                ...messages,
                receivedMessage,
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
