import { OpenAIChatApi } from './src';

(async function go() {
  const openai = new OpenAIChatApi(
    {
      apiKey: process.env.OPENAI_KEY ?? 'YOUR_OPENAI_KEY',
    },
    { contextSize: 4096, model: 'gpt-3.5-turbo-0613' },
  );

  const res = await openai.textCompletion('Hello');
  console.info('Response 1: ', res);

  const res2 = await openai.chatCompletion([
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

  const res3 = await res2.respond({ role: 'user', content: 'testing 123' });
  console.info('Response 3: ', res3);
})();
