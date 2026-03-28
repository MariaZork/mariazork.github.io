import React from 'react';
import Head from 'next/head';
import Header from './Header';
import Footer from './Footer';

interface LayoutProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
  image?: string;
  type?: 'website' | 'article';
}

export default function Layout({
  children,
  title = "Maria Zorkaltseva's Tech Blog",
  description = "Maria Zorkaltseva's tech blog on Mathematics, Statistics, Machine Learning, Deep Learning, and related topics.",
  image = '/images/android-chrome-512x512.png',
  type = 'website',
}: LayoutProps) {
  const siteUrl = 'https://mariazork.github.io';
  const ogImage = image.startsWith('http') ? image : `${siteUrl}${image}`;

  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="description" content={description} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />

        {/* Open Graph */}
        <meta property="og:type" content={type} />
        <meta property="og:title" content={title} />
        <meta property="og:description" content={description} />
        <meta property="og:image" content={ogImage} />
        <meta property="og:site_name" content="Maria Zorkaltseva" />

        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:site" content="@MZorkaltseva" />
        <meta name="twitter:title" content={title} />
        <meta name="twitter:description" content={description} />
        <meta name="twitter:image" content={ogImage} />

        {/* Favicon */}
        <link rel="icon" href="/images/android-chrome-512x512.png" type="image/png" />

        {/* KaTeX */}
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css"
          integrity="sha384-Xi8rHCmBmhbuyyhbI88391ZKP2dmfnOl4rT9ZfRI7mLTdk1wblIUnrIq35nqwEvC"
          crossOrigin="anonymous"
        />
      </Head>

      <div className="flex flex-col min-h-screen bg-surface">
        <Header />
        <main className="flex-grow">
          {children}
        </main>
        <Footer />
      </div>
    </>
  );
}
