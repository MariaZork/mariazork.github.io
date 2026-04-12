import Head from 'next/head';
import { useRouter } from 'next/router';

interface SEOHeadProps {
  title?: string;
  description?: string;
  image?: string;
  type?: 'website' | 'article' | 'profile';
  publishedAt?: string;
  updatedAt?: string;
  author?: string;
  keywords?: string[];
  noindex?: boolean;
  nofollow?: boolean;
  canonical?: string;
  alternateLanguages?: Array<{ hrefLang: string; href: string }>;
  ogImageAlt?: string;
  twitterCardType?: 'summary' | 'summary_large_image' | 'app' | 'player';
}

const SITE_URL = 'https://mariazork.github.io';
const SITE_NAME = 'Maria Zorkaltseva';
const TWITTER_HANDLE = '@MZorkaltseva';
const DEFAULT_DESCRIPTION =
  'Maria Zorkaltseva — Lead Machine Learning Engineer specialising in computer vision, NLP, MLOps, Wi-Fi CSI sensing, and ML systems. Blog, tutorials, and ML interview prep.';
const DEFAULT_IMAGE = `${SITE_URL}/images/android-chrome-512x512.png`;

export default function SEOHead({
  title,
  description = DEFAULT_DESCRIPTION,
  image,
  type = 'website',
  publishedAt,
  updatedAt,
  author = 'Maria Zorkaltseva',
  keywords = [],
  noindex = false,
  nofollow = false,
  canonical,
  alternateLanguages = [],
  ogImageAlt,
  twitterCardType = 'summary_large_image',
}: SEOHeadProps) {
  const router = useRouter();
  const currentPath = router.asPath.split('?')[0];
  const canonicalUrl = canonical || `${SITE_URL}${currentPath}`;
  const ogImage = image ? (image.startsWith('http') ? image : `${SITE_URL}${image}`) : DEFAULT_IMAGE;

  const pageTitle = title ? `${title} | ${SITE_NAME}` : `${SITE_NAME} — ML Engineer, Computer Vision & NLP`;

  const allKeywords = [
    'machine learning engineer',
    'ML engineer',
    'data scientist',
    'computer vision',
    'NLP',
    'LLM',
    'deep learning',
    'PyTorch',
    'MLOps',
    'Paris',
    'Maria Zorkaltseva',
    ...keywords,
  ];

  const robotsContent = [
    noindex ? 'noindex' : 'index',
    nofollow ? 'nofollow' : 'follow',
    'max-snippet:-1',
    'max-image-preview:large',
    'max-video-preview:-1',
  ].join(', ');

  return (
    <Head>
      {/* Basic Meta */}
      <title>{pageTitle}</title>
      <meta name="description" content={description} />
      <meta name="keywords" content={allKeywords.join(', ')} />
      <meta name="author" content={author} />
      <meta name="robots" content={robotsContent} />
      
      {/* Canonical */}
      <link rel="canonical" href={canonicalUrl} />
      
      {/* Alternate Languages */}
      {alternateLanguages.map((alt) => (
        <link key={alt.hrefLang} rel="alternate" hrefLang={alt.hrefLang} href={alt.href} />
      ))}
      <link rel="alternate" hrefLang="x-default" href={canonicalUrl} />

      {/* Open Graph */}
      <meta property="og:type" content={type} />
      <meta property="og:site_name" content={SITE_NAME} />
      <meta property="og:title" content={pageTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:image" content={ogImage} />
      {ogImageAlt && <meta property="og:image:alt" content={ogImageAlt} />}
      <meta property="og:image:width" content="1200" />
      <meta property="og:image:height" content="630" />
      <meta property="og:url" content={canonicalUrl} />
      <meta property="og:locale" content="en_US" />
      
      {/* Article specific OG tags */}
      {type === 'article' && publishedAt && (
        <meta property="article:published_time" content={publishedAt} />
      )}
      {type === 'article' && updatedAt && (
        <meta property="article:modified_time" content={updatedAt} />
      )}
      {type === 'article' && <meta property="article:author" content={author} />}
      {type === 'article' && keywords.length > 0 && (
        <meta property="article:tag" content={keywords.join(', ')} />
      )}

      {/* Twitter Card */}
      <meta name="twitter:card" content={twitterCardType} />
      <meta name="twitter:site" content={TWITTER_HANDLE} />
      <meta name="twitter:creator" content={TWITTER_HANDLE} />
      <meta name="twitter:title" content={pageTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={ogImage} />
      {ogImageAlt && <meta name="twitter:image:alt" content={ogImageAlt} />}

      {/* Icons */}
      <link rel="icon" href="/images/android-chrome-512x512.png" type="image/png" />
      <link rel="apple-touch-icon" href="/images/android-chrome-512x512.png" />
      
      {/* RSS Feed */}
      <link rel="alternate" type="application/rss+xml" title={`${SITE_NAME} Blog RSS`} href={`${SITE_URL}/feed.xml`} />
    </Head>
  );
}
