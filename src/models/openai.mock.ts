import type {
  ModelRequestOptions,
  ModelConfig,
  OpenAIConfig,
  ChatRequestMessage,
  ChatResponse,
} from '../types';

import type { CompletionApi } from './interface';

/**
 * This is the mock implementation of the OpenAIChatApi class.
 * It can be injected onto a function that uses a live instance
 * of OpenAIChatApi, then validate the args that was passed to that instance.
 *
 * Used for testing functions without making live calls
 * to llm providers.
 */
export class MockOpenAIChatApi implements CompletionApi {
  //
  // List of args that the instance has recieved.
  [key: string]: any;
  config: OpenAIConfig;
  modelConfig: ModelConfig;
  chatMessages: ChatRequestMessage[][] = [];
  chatOpt: ModelRequestOptions[] = [];
  textPrompt: string[] = [];
  textOpt: ModelRequestOptions[] = [];
  promptOrMessages: string[][] = [];
  checkProfanityMessage: string[] = [];

  //
  // List of args that the instance is expected to recieve.
  expectedArgs: {
    [key: string]: any;
    constructorArgs?: { config: OpenAIConfig; modelConfig: ModelConfig };
    chatCompletionArgs?: {
      messages: ChatRequestMessage[];
      opt?: ModelRequestOptions;
    }[];
    textCompletionArgs?: { prompt: string; opt?: ModelRequestOptions }[];
    getTokensFromPromptArgs?: { promptOrMessages: string[] }[];
    checkProfanityArgs?: { message: string }[];
  } = {};

  /**
   * The function to set the expected arguments.
   *
   * @param args the expected arguments
   */
  setExpectedArgs(args: this['expectedArgs']) {
    this.expectedArgs = args;
  }

  /**
   * Validate that the arguments recieved match the expected arguments.
   * Might want to return a boolean here, or throw an error.
   * Also, might want to create a validate function for each method instead.
   */
  validateArgs() {
    for (const method in this.expectedArgs) {
      expect(this[method]).toEqual(this.expectedArgs[method]);
    }
  }

  /**
   * The mock implementation of getTokensFromPrompt
   *
   * @param config the config
   * @param modelConfig the model config
   */
  constructor(
    config: OpenAIConfig,
    modelConfig: ModelConfig = { model: 'default' },
  ) {
    this.config = config;
    this.modelConfig = modelConfig;
  }
  /**
   * The mock implementation of chatCompletion
   *
   * @param messages the messages to use
   * @param opt the model request options
   * @returns the mock chat response
   */
  async chatCompletion(
    messages: ChatRequestMessage[],
    opt?: ModelRequestOptions,
  ): Promise<ChatResponse> {
    this.chatMessages.push(messages);
    if (opt) {
      this.chatOpt.push(opt);
    }

    return Promise.resolve({
      content: 'Test Content, this is a chat completion',
      name: 'TestName',
      arguments: {},
      usage: { promptTokens: 10, completionTokens: 20, totalTokens: 30 },
      respond: async () => this.chatCompletion(messages, opt),
    });
  }

  /**
   * The mock implementation of textCompletion
   *
   * @param prompt the prompt to use
   * @param opt the model request options
   * @returns the mock chat response
   */
  async textCompletion(
    prompt: string,
    opt?: ModelRequestOptions,
  ): Promise<ChatResponse> {
    this.textPrompt.push(prompt);
    if (opt) {
      this.textOpt.push(opt);
    }

    return Promise.resolve({
      content: 'Test Content, this is a text completion',
      name: 'TestName',
      arguments: {},
      usage: { promptTokens: 10, completionTokens: 20, totalTokens: 30 },
      respond: async () => this.textCompletion(prompt, opt),
    });
  }

  /**
   * The mock implementation of getTokensFromPrompt
   *
   * @param promptOrMessages the prompt or messages to get tokens from
   * @returns mock number of tokens
   */
  getTokensFromPrompt(promptOrMessages: string[]): number {
    this.promptOrMessages.push(promptOrMessages);
    return -1;
  }
}
