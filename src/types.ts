import Anthropic from '@anthropic-ai/sdk';
import EventEmitter from 'events';
import { ConfigurationParameters as OAIConfig } from 'openai-edge';
import { JsonValue } from 'type-fest';

export type AnthropicConfig = NonNullable<
  ConstructorParameters<typeof Anthropic>[0]
>;

export type OpenAIConfig = OAIConfig & {
  azureEndpoint?: string;
  azureDeployment?: string;
};

export interface ModelConfig {
  model?: string;
  // set this to the total context size of the model, to enable automatic request chunking to avoid context overflows
  contextSize?: number;

  // max tokens to generate
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stop?: string | string[];
  presencePenalty?: number;
  frequencyPenalty?: number;
  logitBias?: Record<string, number>;
  user?: string;

  // use stream mode for API response, the streamed tokens will be sent to `events in `ModelRequestOptions`
  // NOTE: this does NOT support functions
  stream?: boolean;
}

export type ModelFunction = {
  name: string;
  description?: string;
  parameters?: {
    [key: string]: any;
  };
};

export type ModelRequestOptions = {
  systemMessage?: string | (() => string);

  // send a prefix to the model response so the model can continue generating from there, useful for steering the model towards certain output structures.
  // the response prefix WILL be appended to the model response.
  // for Anthropic's models ONLY
  responsePrefix?: string;

  // function related parameters are for OpenAI's models ONLY
  functions?: ModelFunction[];
  // force the model to call the following function
  callFunction?: string;

  // the number of time to retry this request due to rate limit or recoverable API errors
  retries?: number;
  retryInterval?: number;
  timeout?: number;

  // the minimum amount of tokens to allocate for the response. if the request is predicted to not have enough tokens, it will automatically throw a 'TokenError' without sending the request
  minimumResponseTokens?: number;

  // the maximum amount of tokens to use for response
  // NOTE: in OpenAI models, setting this option also requires contextSize in ModelConfig to be set
  maximumResponseTokens?: number;

  // pass in an event emitter to receive message stream events
  events?: EventEmitter;
};

export type ChatRequestRole = 'system' | 'user' | 'assistant' | 'function';

export interface ChatRequestMessage {
  role: ChatRequestRole;
  content?: string;
  name?: string;
  function_call?: ChatRequestFunctionCall;
}

export interface ChatRequestFunctionCall {
  name?: string;
  arguments?: string;
}

export type ChatResponse = {
  content?: string;

  // name and argument used for function reponse
  name?: string;
  arguments?: JsonValue;

  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };

  // function to send another message in the same chat, this will automatically reuse all existing settings, and append a new message to the messages array
  respond: (
    message: string | ChatRequestMessage,
    opt?: ModelRequestOptions,
  ) => Promise<ChatResponse>;
};
