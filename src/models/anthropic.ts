import Anthropic from '@anthropic-ai/sdk';
import { MessageCreateParamsBase } from '@anthropic-ai/sdk/resources/messages';
import { compact, defaults } from 'lodash';

import {
  CompletionDefaultRetries,
  CompletionDefaultTimeout,
  DefaultAnthropicModel,
  MaximumResponseTokens,
  MinimumResponseTokens,
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
  timeout: CompletionDefaultTimeout,
  minimumResponseTokens: MinimumResponseTokens,
  maximumResponseTokens: MaximumResponseTokens,
};

export class AnthropicChatApi implements CompletionApi {
  client: Anthropic;
  modelConfig: ModelConfig;

  constructor(config?: AnthropicConfig, modelConfig?: ModelConfig) {
    this.client = new Anthropic(config);
    this.modelConfig = modelConfig ?? {};
  }

  getTokensFromPrompt = getTikTokenTokensFromPrompt;

  // chat based prompting following these instructions:
  // https://docs.anthropic.com/claude/reference/getting-started-with-the-api
  async chatCompletion(
    initialMessages: ChatRequestMessage[],
    requestOptions?: ModelRequestOptions | undefined,
  ): Promise<ChatResponse> {
    const finalRequestOptions = defaults(requestOptions, RequestDefaults);
    const messages: ChatRequestMessage[] = compact([
      ...initialMessages,
      // claude supports responsePrefix via prefill:
      // https://docs.anthropic.com/claude/docs/prefill-claudes-response
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
    const completionBody: MessageCreateParamsBase = {
      stop_sequences:
        typeof finalRequestOptions.stop === 'string'
          ? [finalRequestOptions.stop]
          : finalRequestOptions.stop,
      temperature: this.modelConfig.temperature,
      top_p: this.modelConfig.topP,
      model: this.modelConfig.model ?? DefaultAnthropicModel,
      max_tokens: finalRequestOptions.maximumResponseTokens,
      system: finalRequestOptions.systemMessage
        ? typeof finalRequestOptions.systemMessage === 'string'
          ? finalRequestOptions.systemMessage
          : finalRequestOptions.systemMessage()
        : undefined,
      // anthropic only supports user and assistant messages, filter all other ones out
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
      const stream = await this.client.messages.stream(
        completionBody,
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
        if (
          part.type === 'content_block_start' &&
          part.content_block.type === 'text' &&
          part.index === 0
        ) {
          const text = part.content_block.text;
          debug.write(text);
          completion += text;
          finalRequestOptions?.events?.emit('data', text);
        } else if (
          part.type === 'content_block_delta' &&
          part.delta.type === 'text_delta' &&
          part.index === 0
        ) {
          const text = part.delta.text;
          debug.write(text);
          completion += text;
          finalRequestOptions?.events?.emit('data', text);
        }
      }

      debug.write('\n[STREAM] response end\n');
    } else {
      const response = await this.client.messages.create(
        completionBody,
        completionOptions,
      );

      if ('content' in response) {
        completion = response.content[0].text;
        debug.log('🔽 completion received', completion);
      }
    }

    const content = finalRequestOptions.responsePrefix
      ? finalRequestOptions.responsePrefix + completion
      : // if no prefix, process the completion a bit by trimming since claude tends to output an extra white space at the beginning
        completion.trim();
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
            // don't send the processed `messages` array, since that contains the prefill message which will cause multiple 'assistant' message error to be thrown
            ...initialMessages,
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
