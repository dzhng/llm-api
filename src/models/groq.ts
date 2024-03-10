import { Groq } from 'groq-sdk';
import { ChatCompletionCreateParamsBase } from 'groq-sdk/resources/chat/completions';
import { compact, defaults } from 'lodash';

import {
  CompletionDefaultRetries,
  CompletionDefaultTimeout,
  DefaultGroqModel,
  MaximumResponseTokens,
  MinimumResponseTokens,
} from '../config';
import {
  ChatRequestMessage,
  ModelConfig,
  ModelRequestOptions,
  ChatResponse,
  GroqConfig,
} from '../types';
import { debug } from '../utils';

import { TokenError } from './errors';
import { CompletionApi } from './interface';
import { getTikTokenTokensFromPrompt } from './tokenizer';

const RequestDefaults = {
  retries: CompletionDefaultRetries,
  timeout: CompletionDefaultTimeout,
  minimumResponseTokens: MinimumResponseTokens,
  maximumResponseTokens: MaximumResponseTokens,
};

export class GroqChatApi implements CompletionApi {
  client: Groq;
  modelConfig: ModelConfig;

  constructor(config?: GroqConfig, modelConfig?: ModelConfig) {
    this.client = new Groq(config);
    this.modelConfig = modelConfig ?? {};
  }

  getTokensFromPrompt = getTikTokenTokensFromPrompt;

  async chatCompletion(
    initialMessages: ChatRequestMessage[],
    requestOptions?: ModelRequestOptions | undefined,
  ): Promise<ChatResponse> {
    const finalRequestOptions = defaults(requestOptions, RequestDefaults);
    const messages: ChatRequestMessage[] = compact([
      finalRequestOptions.systemMessage
        ? {
            role: 'system',
            content:
              typeof finalRequestOptions.systemMessage === 'string'
                ? finalRequestOptions.systemMessage
                : finalRequestOptions.systemMessage(),
          }
        : null,
      ...initialMessages,
      finalRequestOptions.responsePrefix
        ? ({
            role: 'assistant',
            content: finalRequestOptions.responsePrefix,
          } as ChatRequestMessage)
        : null,
    ]);

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
    );
    if (messageTokens > maxPromptTokens) {
      throw new TokenError(
        'Prompt too big, not enough tokens to meet minimum response',
        messageTokens - maxPromptTokens,
      );
    }

    let completion = '';
    const completionBody: ChatCompletionCreateParamsBase = {
      stop: finalRequestOptions.stop,
      temperature: this.modelConfig.temperature,
      top_p: this.modelConfig.topP,
      model: this.modelConfig.model ?? DefaultGroqModel,
      max_tokens: finalRequestOptions.maximumResponseTokens,
      // filter all other messages except user and assistant ones
      messages: messages
        .filter(
          (m) => (m.role === 'user' || m.role === 'assistant') && m.content,
        )
        .map((m) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content ?? '',
        })),
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

      for await (const part of stream) {
        const text = part.choices[0]?.delta?.content ?? '';
        debug.write(text);
        completion += text;
        finalRequestOptions?.events?.emit('data', text);
      }

      debug.write('\n[STREAM] response end\n');
    } else {
      const response = await this.client.chat.completions.create(
        { ...completionBody, stream: false },
        completionOptions,
      );
      completion = response.choices[0].message.content ?? '';
      debug.log('🔽 completion received', completion);
    }

    const content = finalRequestOptions.responsePrefix
      ? finalRequestOptions.responsePrefix + completion
      : completion;
    if (!content) {
      throw new Error('Completion response malformed');
    }

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
    };
  }

  textCompletion(
    prompt: string,
    requestOptions = {} as Partial<ModelRequestOptions>,
  ): Promise<ChatResponse> {
    const messages: ChatRequestMessage[] = [{ role: 'user', content: prompt }];
    return this.chatCompletion(messages, requestOptions);
  }
}
