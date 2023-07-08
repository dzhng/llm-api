# ✨ LLM API

[![test](https://github.com/dzhng/llm-api/actions/workflows/test.yml/badge.svg?branch=main&event=push)](https://github.com/dzhng/llm-api/actions/workflows/test.yml)

Fully typed chat APIs for OpenAI and Azure's chat models for browser, edge, and node environments.

- [Introduction](#-introduction)
- [Usage](#-usage)
- [Azure](#-azure)
- [Debugging](#-debugging)

## 👋 Introduction

- Clean interface for text and chat completion for OpenAI and Azure models
- Catch token overflow errors automatically on the client side
- Handle rate limit and any other API errors as gracefully as possible (e.g. exponential backoff for rate-limit)
- Support for browser, edge, and node environments

```typescript
import { OpenAIChatApi } from 'llm-api';

const openai = new OpenAIChatApi({ apiKey: 'YOUR_OPENAI_KEY' });

const resText = await openai.textCompletion('Hello');

const resChat = await openai.chatCompletion({
  role: 'user',
  content: 'Hello world',
});
```

## 🔨 Usage

### Install

This package is hosted on npm:

```
npm i llm-api
```

```
yarn add llm-api
```

### Model Config

To configure a new model endpoint:

```typescript
const openai = new OpenAIChatApi(params: OpenAIConfig, config: ModelConfig);
const anthropic = new AnthropicChatApi(params: AnthropicConfig, config: ModelConfig);
```

These model config map to OpenAI's config directly, see doc:
https://platform.openai.com/docs/api-reference/chat/create

```typescript
interface ModelConfig {
  model?: string;
  contextSize?: number;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stop?: string | string[];
  presencePenalty?: number;
  frequencyPenalty?: number;
  logitBias?: Record<string, number>;
  user?: string;
}
```

### Request

To send a completion request to a model:

```typescript
const text: ModelResponse = await openai.textCompletion(api: CompletionApi, prompt: string, options: ModelRequestOptions);

const completion: ModelResponse = await openai.chatCompletion(api: CompletionApi, messages: ChatCompletionRequestMessage, options: ModelRequestOptions);

// respond to existing chat session, preserving the past messages
const response: ModelResponse = await completion.respond(message: ChatCompletionRequestMessage, options: ModelRequestOptions);
```

**options**
You can override the default request options via this parameter. A request will automatically be retried if there is a ratelimit or server error.

```typescript
type ModelRequestOptions = {
  // set to automatically add system message (only relevant when using textCompletion)
  systemMessage?: string | (() => string);

  // function to pass into context on OpenAI's new 0613 models
  functions?: ModelFunction[];

  // default: 3
  retries?: number;
  // default: 30s
  retryInterval?: number;
  // default: 60s
  timeout?: number;

  // the minimum amount of tokens to allocate for the response. if the request is predicted to not have enough tokens, it will automatically throw a 'TokenError' without sending the request
  // default: 200
  minimumResponseTokens?: number;
};
```

### Response

Completion responses are in the following format:

```typescript
interface ModelResponse {
  content?: string;

  // used to parse function responses
  name?: string;
  arguments?: string;

  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };

  // function to send another message in the same 'chat', this will automatically append a new message to the messages array
  respond: (
    message: ChatCompletionRequestMessage,
    opt?: ModelRequestOptions,
  ) => Promise<ModelResponse>;
}
```

### 📃 Token Errors

A common error with LLM APIs is token usage - you are only allowed to fit a certain amount of data in the context window.

If you set a `contextSize` key, `llm-api` will automatically determine if the request will breach the token limit BEFORE sending the actual request to the model provider (e.g. OpenAI). This will save one network round-trip call and let you handle these type of errors in a responsive manner.

```typescript
const openai = new OpenAIChatApi(
  { apiKey: 'YOUR_OPENAI_KEY' },
  { model: 'gpt-4-0613', contextSize: 8129 },
);

try {
  const res = await openai.textCompletion(...);
} catch (e) {
  if (e instanceof TokenError) {
    // handle token errors...
  }
}
```

## 🔷 Azure

`llm-api` also comes with support for Azure's OpenAI models. The Azure version is usually much faster and more reliable than OpenAI's own API endpoints. In order to use the Azure endpoints, you must include 2 Azure specific options when initializing the OpenAI model, `azureDeployment` and `azureEndpoint`. The `apiKey` field will also now be used for the Azure API key.

You can find the Azure API key and endpoint in the [Azure Portal](https://portal.azure.com/). The Azure Deployment must be created under the [Azure AI Portal](https://oai.azure.com/).

Note that the `model` parameter in `ModelConfig` will be ignored when using Azure. This is because in the Azure system, the `model` is selected on deployment creation, not on run time.

```typescript
const model = new OpenAIChatApi({
  apiKey: 'AZURE_OPENAI_KEY',
  azureDeployment: 'AZURE_DEPLOYMENT_NAME',
  azureEndpoint: 'AZURE_ENDPOINT',
});
```

## 🤓 Debugging

`llm-api` usese the `debug` module for logging & error messages. To run in debug mode, set the `DEBUG` env variable:

`DEBUG=llm-api:* yarn playground`

You can also specify different logging types via:

`DEBUG=llm-api:error yarn playground`
`DEBUG=llm-api:log yarn playground`
