import { debug as mDebug } from 'debug';
import jsonic from 'jsonic';
import { jsonrepair } from 'jsonrepair';

const error = mDebug('llm-api:error');
const log = mDebug('llm-api:log');
// eslint-disable-next-line no-console
log.log = console.log.bind(console);

export const debug = {
  error,
  log,
  write: (t: string) =>
    process.env.DEBUG &&
    'llm-api:log'.match(process.env.DEBUG) &&
    process.stdout.write(t),
};

export function sleep(delay: number) {
  return new Promise((resolve) => {
    setTimeout(resolve, delay);
  });
}

export function parseUnsafeJson(json: string): any {
  return jsonic(jsonrepair(json));
}

export type MaybePromise<T> = Promise<T> | T;
