import Anthropic, { AI_PROMPT, HUMAN_PROMPT } from '@anthropic-ai/sdk';
import { compact, defaults } from 'lodash';

import {
  CompletionDefaultRetries,
  CompletionDefaultTimeout,
  DefaultAnthropicModel,
  MinimumResponseTokens,
  RateLimitRetryIntervalMs,
} from '../config';
import {
  AnthropicConfig,
  ChatRequestMessage,
  ModelConfig,
  ModelRequestOptions,
  ChatResponse,
} from '../types';
import { debug } from '../utils';

import { TokenError } from './errors';
import { CompletionApi } from './interface';
import { getTikTokenTokensFromPrompt } from './tokenizer';

const RequestDefaults = {
  retries: CompletionDefaultRetries,
  retryInterval: RateLimitRetryIntervalMs,
  timeout: CompletionDefaultTimeout,
  minimumResponseTokens: MinimumResponseTokens,
};

export class AnthropicChatApi implements CompletionApi {
  _client: Anthropic;
  modelConfig: ModelConfig;

  constructor(config?: AnthropicConfig, modelConfig?: ModelConfig) {
    this._client = new Anthropic(config);
    this.modelConfig = modelConfig ?? {};
  }

  getTokensFromPrompt = getTikTokenTokensFromPrompt;

  // chat based prompting following these instructions:
  // https://docs.anthropic.com/claude/reference/getting-started-with-the-api
  async chatCompletion(
    messages: ChatRequestMessage[],
    requestOptions?: ModelRequestOptions | undefined,
  ): Promise<ChatResponse> {
    const finalRequestOptions = defaults(requestOptions, RequestDefaults);
    const prompt =
      messages
        .map((message) => {
          switch (message.role) {
            case 'user':
              return `${HUMAN_PROMPT} ${message.content}`;
            case 'assistant':
              return `${AI_PROMPT} ${message.content}`;
            case 'system':
              return `${HUMAN_PROMPT} ${message.content}`;
            default:
              throw new Error(
                `Anthropic models do not support message with the role ${message.role}`,
              );
          }
        })
        .join('') + AI_PROMPT;

    debug.log(
      `🔼 completion requested:\n${prompt}\nconfig: ${JSON.stringify(
        this.modelConfig,
      )}, options: ${JSON.stringify(finalRequestOptions)}`,
    );

    // check if we'll have enough tokens to meet the minimum response
    const maxPromptTokens = this.modelConfig.contextSize
      ? this.modelConfig.contextSize - finalRequestOptions.minimumResponseTokens
      : 1_000_000;

    const messageTokens = this.getTokensFromPrompt([prompt]);
    if (messageTokens > maxPromptTokens) {
      throw new TokenError(
        'Prompt too big, not enough tokens to meet minimum response',
        messageTokens - maxPromptTokens,
      );
    }

    const response = await this._client.completions.create(
      {
        stop_sequences:
          typeof this.modelConfig.stop === 'string'
            ? [this.modelConfig.stop]
            : this.modelConfig.stop,
        max_tokens_to_sample: finalRequestOptions.minimumResponseTokens,
        temperature: this.modelConfig.temperature,
        top_p: this.modelConfig.topP,
        model: this.modelConfig.model ?? DefaultAnthropicModel,
        prompt,
      },
      {
        timeout: finalRequestOptions.timeout,
        maxRetries: finalRequestOptions.retries,
      },
    );

    const content = response.completion.trim();
    if (!content) {
      throw new Error('Completion response malformed');
    }

    return {
      content,
      respond: (message: ChatRequestMessage, opt) =>
        this.chatCompletion(
          [...messages, { role: 'assistant', content }, message],
          opt,
        ),
    };
  }

  textCompletion(
    prompt: string,
    requestOptions = {} as Partial<ModelRequestOptions>,
  ): Promise<ChatResponse> {
    const messages: ChatRequestMessage[] = compact([
      requestOptions.systemMessage
        ? {
            role: 'system',
            content:
              typeof requestOptions.systemMessage === 'string'
                ? requestOptions.systemMessage
                : requestOptions.systemMessage(),
          }
        : undefined,
      { role: 'user', content: prompt },
    ]);
    return this.chatCompletion(messages, requestOptions);
  }
}
