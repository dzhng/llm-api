import { AI_PROMPT, HUMAN_PROMPT } from '@anthropic-ai/sdk';
import {
  BedrockRuntimeClient,
  BedrockRuntimeClientConfig,
  InvokeModelCommand,
  InvokeModelCommandInput,
} from '@aws-sdk/client-bedrock-runtime';
import { defaults } from 'lodash';

import {
  CompletionDefaultRetries,
  CompletionDefaultTimeout,
  MaximumResponseTokens,
  MinimumResponseTokens,
  RateLimitRetryIntervalMs,
} from '../config';
import {
  ChatRequestMessage,
  ChatResponse,
  ModelConfig,
  ModelRequestOptions,
} from '../types';

import { TokenError } from './errors';
import { CompletionApi } from './interface';
import { getTikTokenTokensFromPrompt } from './tokenizer';

const ForbiddenTokens = [HUMAN_PROMPT.trim(), AI_PROMPT.trim()];

const RequestDefaults = {
  retries: CompletionDefaultRetries,
  retryInterval: RateLimitRetryIntervalMs,
  timeout: CompletionDefaultTimeout,
  minimumResponseTokens: MinimumResponseTokens,
  maximumResponseTokens: MaximumResponseTokens,
};

export class AnthropicBedrockChat implements CompletionApi {
  modelConfig: ModelConfig;
  _client: BedrockRuntimeClient;

  constructor(
    config: BedrockRuntimeClientConfig['credentials'],
    modelConfig?: ModelConfig,
  ) {
    this.modelConfig = modelConfig ?? {};

    this._client = new BedrockRuntimeClient({
      region: 'us-east-1',
      serviceId: 'bedrock-runtime',
      credentials: config,
    });
  }

  async chatCompletion(
    initialMessages: ChatRequestMessage[],
    requestOptions?: ModelRequestOptions | undefined,
  ): Promise<ChatResponse> {
    const finalRequestOptions = defaults(requestOptions, RequestDefaults);
    const messages: ChatRequestMessage[] = (
      finalRequestOptions.systemMessage
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
        : initialMessages
    ).map(
      (message) =>
        ({
          ...message,
          // automatically remove forbidden tokens in the input message to thwart prompt injection attacks
          content:
            message.content &&
            ForbiddenTokens.reduce(
              (prev, token) => prev.replaceAll(token, ''),
              message.content,
            ),
        } as ChatRequestMessage),
    );

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
        .join('') +
      AI_PROMPT +
      (finalRequestOptions.responsePrefix
        ? ` ${finalRequestOptions.responsePrefix}`
        : '');

    const maxPromptTokens = this.modelConfig.contextSize
      ? this.modelConfig.contextSize - finalRequestOptions.minimumResponseTokens
      : 100_000;

    const messageTokens = this.getTokensFromPrompt([prompt]);
    if (messageTokens > maxPromptTokens) {
      throw new TokenError(
        'Prompt too big, not enough tokens to meet minimum response',
        messageTokens - maxPromptTokens,
      );
    }

    const params: InvokeModelCommandInput = {
      modelId: this.modelConfig.model || 'anthropic.claude-v2',
      contentType: 'application/json',
      accept: '*/*',
      body: JSON.stringify({
        prompt,
        max_tokens_to_sample: finalRequestOptions.maximumResponseTokens,
        temperature: this.modelConfig.temperature,
        top_p: this.modelConfig.topP || 1,
        stop_sequences:
          typeof this.modelConfig.stop === 'string'
            ? [this.modelConfig.stop]
            : this.modelConfig.stop,
        anthropic_version: 'bedrock-2023-05-31',
      }),
    };

    const completionOptions = {
      requestTimeout: finalRequestOptions.timeout,
      //abortSignal
    };

    const command = new InvokeModelCommand(params);
    const response = await this._client.send(command, completionOptions);
    const content = new TextDecoder().decode(response.body);

    // TODO add logic for streaming response

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
    };
  }
  textCompletion(
    prompt: string,
    requestOptions = {} as Partial<ModelRequestOptions>,
  ): Promise<ChatResponse> {
    const messages: ChatRequestMessage[] = [{ role: 'user', content: prompt }];
    return this.chatCompletion(messages, requestOptions);
  }

  getTokensFromPrompt = getTikTokenTokensFromPrompt;
}
