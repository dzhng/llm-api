import { OpenAIChatApi } from './src';

(async function go() {
  const openai = new OpenAIChatApi(
    {
      apiKey: process.env.OPENAI_KEY ?? 'YOUR_OPENAI_KEY',
    },
    { contextSize: 4096, model: 'gpt-3.5-turbo-0613' },
  );

  const res = await openai.textCompletion('Hello');
  console.info('Response: ', res);
})();
