import {
  ModelRequestOptions,
  ChatResponse,
  ModelConfig,
  ChatRequestMessage,
} from '../types';

export interface CompletionApi {
  modelConfig: ModelConfig;

  chatCompletion(
    messages: ChatRequestMessage[],
    opt?: ModelRequestOptions,
  ): Promise<ChatResponse>;

  textCompletion(
    prompt: string,
    opt?: ModelRequestOptions,
  ): Promise<ChatResponse>;

  getTokensFromPrompt(promptOrMessages: string[]): number;
}
