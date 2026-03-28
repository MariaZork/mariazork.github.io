import type { AppProps } from 'next/app';
import '@/styles/globals.css';
import 'katex/dist/katex.min.css';
import '@/styles/katex-custom.css';
import "highlight.js/styles/github-dark.css";

export default function MyApp({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />;
}
