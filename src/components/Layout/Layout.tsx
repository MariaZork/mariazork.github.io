import Head from 'next/head';
import { useRouter } from 'next/router';
import Header from './Header';
import Footer from './Footer';

interface LayoutProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
  image?: string;
  type?: 'website' | 'article';
  publishedAt?: string;
  updatedAt?: string;
  author?: string;
  keywords?: string[];
}

const SITE_URL = 'https://mariazork.github.io';
const SITE_NAME = 'Maria Zorkaltseva';
const TWITTER_HANDLE = '@MZorkaltseva';
const DEFAULT_DESCRIPTION =
  'Maria Zorkaltseva — Lead Machine Learning Engineer specialising in computer vision, NLP, MLOps, Wi-Fi CSI sensing, and ML systems. Blog, tutorials, and ML interview prep.';
const DEFAULT_IMAGE = `${SITE_URL}/images/android-chrome-512x512.png`;

export default function Layout({
  children,
  title,
  description = DEFAULT_DESCRIPTION,
  image,
  type = 'website',
  publishedAt,
  updatedAt,
  author = 'Maria Zorkaltseva',
  keywords = [],
}: LayoutProps) {
  const router = useRouter();
  const canonical = `${SITE_URL}${router.asPath.split('?')[0]}`;
  const ogImage = image
    ? image.startsWith('http') ? image : `${SITE_URL}${image}`
    : DEFAULT_IMAGE;

  const pageTitle = title
    ? `${title} | ${SITE_NAME}`
    : `${SITE_NAME} — ML Engineer, Computer Vision & NLP`;

  const allKeywords = [
    'machine learning engineer',
    'computer vision',
    'NLP',
    'deep learning',
    'PyTorch',
    'MLOps',
    'Paris',
    'Maria Zorkaltseva',
    ...keywords,
  ];

  const jsonLd = type === 'article'
    ? {
        '@context': 'https://schema.org',
        '@type': 'TechArticle',
        headline: title,
        description,
        image: ogImage,
        author: {
          '@type': 'Person',
          name: author,
          url: SITE_URL,
          sameAs: [
            'https://github.com/MariaZork',
            'https://www.linkedin.com/in/maria-zorkaltseva/',
            'https://scholar.google.com/citations?user=kJHS8ygAAAAJ',
          ],
        },
        publisher: { '@type': 'Person', name: SITE_NAME, url: SITE_URL },
        url: canonical,
        ...(publishedAt && { datePublished: publishedAt }),
        ...(updatedAt && { dateModified: updatedAt }),
        ...(keywords.length > 0 && { keywords: keywords.join(', ') }),
        inLanguage: 'en',
        mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
      }
    : {
        '@context': 'https://schema.org',
        '@type': 'WebPage',
        name: pageTitle,
        description,
        url: canonical,
        author: {
          '@type': 'Person',
          name: SITE_NAME,
          jobTitle: 'Lead Machine Learning Engineer',
          url: SITE_URL,
          email: 'maria.zorkaltseva@gmail.com',
          address: { '@type': 'PostalAddress', addressLocality: 'Paris', addressCountry: 'FR' },
          sameAs: [
            'https://github.com/MariaZork',
            'https://www.linkedin.com/in/maria-zorkaltseva/',
            'https://twitter.com/MZorkaltseva',
            'https://scholar.google.com/citations?user=kJHS8ygAAAAJ',
          ],
          knowsAbout: ['Machine Learning', 'Computer Vision', 'NLP', 'MLOps', 'Deep Learning', 'PyTorch', 'Signal Processing'],
        },
      };

  return (
    <>
      <Head>
        <title>{pageTitle}</title>
        <meta name="description" content={description} />
        <meta name="keywords" content={allKeywords.join(', ')} />
        <meta name="author" content={author} />
        <meta name="robots" content="index, follow, max-snippet:-1, max-image-preview:large, max-video-preview:-1" />
        <link rel="canonical" href={canonical} />

        <meta property="og:type" content={type} />
        <meta property="og:site_name" content={SITE_NAME} />
        <meta property="og:title" content={pageTitle} />
        <meta property="og:description" content={description} />
        <meta property="og:image" content={ogImage} />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
        <meta property="og:url" content={canonical} />
        <meta property="og:locale" content="en_US" />
        {type === 'article' && publishedAt && (
          <meta property="article:published_time" content={publishedAt} />
        )}
        {type === 'article' && updatedAt && (
          <meta property="article:modified_time" content={updatedAt} />
        )}
        {type === 'article' && (
          <meta property="article:author" content={author} />
        )}

        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:site" content={TWITTER_HANDLE} />
        <meta name="twitter:creator" content={TWITTER_HANDLE} />
        <meta name="twitter:title" content={pageTitle} />
        <meta name="twitter:description" content={description} />
        <meta name="twitter:image" content={ogImage} />

        <link rel="icon" href="/images/android-chrome-512x512.png" type="image/png" />
        <link rel="apple-touch-icon" href="/images/android-chrome-512x512.png" />
        <link rel="alternate" type="application/rss+xml" title={`${SITE_NAME} Blog RSS`} href={`${SITE_URL}/feed.xml`} />

        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css"
          integrity="sha384-Xi8rHCmBmhbuyyhbI88391ZKP2dmfnOl4rT9ZfRI7mLTdk1wblIUnrIq35nqwEvC"
          crossOrigin="anonymous"
        />

        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </Head>

      <div className="flex flex-col min-h-screen bg-surface">
        <Header />
        <main className="flex-grow" id="main-content">
          {children}
        </main>
        <Footer />
      </div>
    </>
  );
}