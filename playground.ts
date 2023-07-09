import { OpenAIChatApi, AnthropicChatApi } from './src';

(async function go() {
  const client = process.env.OPENAI_KEY
    ? new OpenAIChatApi(
        {
          apiKey: process.env.OPENAI_KEY ?? 'YOUR_client_KEY',
        },
        { contextSize: 4096, model: 'gpt-3.5-turbo-0613' },
      )
    : process.env.ANTHROPIC_KEY
    ? new AnthropicChatApi({
        apiKey: process.env.ANTHROPIC_KEY ?? 'YOUR_client_KEY',
      })
    : undefined;

  const res0 = await client?.textCompletion('Hello', {
    systemMessage: 'You will respond to all human messages in JSON',
    responsePrefix: '{ "message": "',
  });
  console.info('Response 0: ', res0);

  const res1 = await client?.textCompletion('Hello', {
    maximumResponseTokens: 1,
  });
  console.info('Response 1: ', res1);

  const res2 = await client?.chatCompletion([
    { role: 'user', content: 'hello' },
    {
      role: 'assistant',
      content: '',
      function_call: {
        name: 'print',
        arguments: '{"hello": "world"}',
      },
    },
  ]);
  console.info('Response 2: ', res2);

  const res3 = await res2?.respond({ role: 'user', content: 'testing 123' });
  console.info('Response 3: ', res3);
})();
