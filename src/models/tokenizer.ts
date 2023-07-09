import tiktoken from 'js-tiktoken';

import { ModelFunction } from '../types';

const encoder = tiktoken.getEncoding('cl100k_base');

export function getTikTokenTokensFromPrompt(
  promptOrMessages: string[],
  functions?: ModelFunction[],
) {
  let numTokens = 0;

  for (const message of promptOrMessages) {
    numTokens += 5; // every message follows <im_start>{role/name}\n{content}<im_end>\n
    numTokens += encoder.encode(message).length;
  }
  numTokens += 2; // every reply is primed with <im_start>assistant\n

  if (functions) {
    for (const func of functions) {
      numTokens += 5;
      numTokens += encoder.encode(JSON.stringify(func)).length;
    }
    // estimate tokens needed to prime functions
    numTokens += 20;
  }

  return numTokens;
}
