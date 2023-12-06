// legacy openai implementation using function calls instead of tool calls
import 'openai/shims/web';
import { defaults } from 'lodash';
import { OpenAI } from 'openai';
import type { CompletionUsage } from 'openai/resources';
import type { ChatCompletionCreateParamsBase } from 'openai/resources/chat/completions';

import {
  CompletionDefaultRetries,
  CompletionDefaultTimeout,
  DefaultAzureVersion,
  DefaultOpenAIModel,
  MinimumResponseTokens,
} from '../config';
import type {
  ModelRequestOptions,
  ModelConfig,
  OpenAIConfig,
  ChatRequestMessage,
  ChatResponse,
} from '../types';
import { debug, parseUnsafeJson } from '../utils';

import { TokenError } from './errors';
import type { CompletionApi } from './interface';
import { getTikTokenTokensFromPrompt } from './tokenizer';

const RequestDefaults = {
  retries: CompletionDefaultRetries,
  timeout: CompletionDefaultTimeout,
  minimumResponseTokens: MinimumResponseTokens,
  // NOTE: this is left without defaults by design - OpenAI's API will throw an error if max_token values is greater than model context size, which means it needs to be different for every model and cannot be set as a default. This fine since OpenAI won't put any limit on max_tokens if it's not set anyways (unlike Anthropic).
  // maximumResponseTokens: MaximumResponseTokens,
};

const convertConfig = (
  config: Partial<ModelConfig>,
): Partial<ChatCompletionCreateParamsBase> => ({
  model: config.model,
  temperature: config.temperature,
  top_p: config.topP,
  n: 1,
  presence_penalty: config.presencePenalty,
  frequency_penalty: config.frequencyPenalty,
  logit_bias: config.logitBias,
  user: config.user,
  stream: config.stream,
});

export class OpenAILegacyChatApi implements CompletionApi {
  client: OpenAI;
  _isAzure: boolean;
  _headers?: Record<string, string>;
  modelConfig: ModelConfig;

  constructor(config: OpenAIConfig, modelConfig?: ModelConfig) {
    this._isAzure = Boolean(config.azureEndpoint && config.azureDeployment);
    this.client = new OpenAI({
      ...config,
      baseURL: this._isAzure
        ? `${config.azureEndpoint}${
            config.azureEndpoint?.at(-1) === '/' ? '' : '/'
          }openai/deployments/${config.azureDeployment}`
        : undefined,
      defaultHeaders: this._isAzure
        ? { 'api-key': String(config.apiKey) }
        : undefined,
      defaultQuery: this._isAzure
        ? {
            'api-version': config.azureApiVersion ?? DefaultAzureVersion,
          }
        : undefined,
    });

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

    // check if we'll have enough tokens to meet the minimum response
    const maxPromptTokens = this.modelConfig.contextSize
      ? this.modelConfig.contextSize - finalRequestOptions.minimumResponseTokens
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

    // calculate max response tokens
    // note that for OpenAI models, it MUST be conditional on the contextSize being set, this is because OpenAI's API throws an error if maxTokens is above context size
    const maxTokens =
      this.modelConfig.contextSize && finalRequestOptions.maximumResponseTokens
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

    let completion = '';
    let functionCall: { name: string; arguments: string } | undefined;
    let usage: CompletionUsage | undefined;
    const completionBody: ChatCompletionCreateParamsBase = {
      model: DefaultOpenAIModel,
      ...convertConfig(this.modelConfig),
      max_tokens: maxTokens,
      stop: finalRequestOptions.stop,
      functions: finalRequestOptions.functions,
      function_call: finalRequestOptions.callFunction
        ? { name: finalRequestOptions.callFunction }
        : finalRequestOptions.functions
        ? 'auto'
        : undefined,
      messages: messages.map((m) =>
        m.role === 'assistant'
          ? {
              role: 'assistant',
              content: m.content ?? null,
              function_call: m.toolCall?.function,
            }
          : m.role === 'tool'
          ? {
              role: 'user', // NOTE: legacy models dont have tool role, so just return user role
              content: m.content ?? null,
            }
          : {
              role: m.role,
              content: m.content ?? null,
            },
      ),
    };
    const completionOptions = {
      timeout: finalRequestOptions.timeout,
      maxRetries: finalRequestOptions.retries,
    };

    if (this.modelConfig.stream) {
      const stream = await this.client.chat.completions.create(
        { ...completionBody, stream: true },
        completionOptions,
      );

      // emit prefix since technically that's counted as part of the response
      if (finalRequestOptions?.responsePrefix) {
        finalRequestOptions?.events?.emit(
          'data',
          finalRequestOptions.responsePrefix,
        );
      }

      const functionCallStreamParts: Partial<
        NonNullable<typeof functionCall>
      >[] = [];
      for await (const part of stream) {
        const text = part.choices[0]?.delta?.content;
        const call = part.choices[0]?.delta?.function_call as Partial<
          typeof functionCall
        >;
        if (text) {
          debug.write(text);
          completion += text;
          finalRequestOptions?.events?.emit('data', text);
        } else if (call) {
          debug.write(
            call.name
              ? `${call.name}: ${call.arguments}\n`
              : call.arguments ?? '',
          );
          functionCallStreamParts.push(call);
        }
      }

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

      debug.write('\n[STREAM] response end\n');
    } else {
      const response = await this.client.chat.completions.create(
        { ...completionBody, stream: false },
        completionOptions,
      );
      completion = response.choices[0].message.content ?? '';
      functionCall = response.choices[0].message.function_call;
      usage = response.usage;
      debug.log('🔽 completion received', completion);
    }

    if (completion) {
      const receivedMessage: ChatRequestMessage = {
        role: 'assistant',
        content: completion,
      };
      return {
        message: receivedMessage,
        content: completion,
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
        toolCall: { type: 'function', id: '1', function: functionCall }, // NOTE: id is redundant here since legacy type won't use it
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
  }

  async textCompletion(
    prompt: string,
    requestOptions = {} as Partial<ModelRequestOptions>,
  ): Promise<ChatResponse> {
    const messages: ChatRequestMessage[] = [{ role: 'user', content: prompt }];
    return this.chatCompletion(messages, requestOptions);
  }
}
