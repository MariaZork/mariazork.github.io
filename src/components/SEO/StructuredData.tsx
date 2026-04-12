import React from 'react';

interface BreadcrumbItem {
  name: string;
  item: string;
}

interface BreadcrumbDataProps {
  items: BreadcrumbItem[];
}

export function BreadcrumbData({ items }: BreadcrumbDataProps) {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: item.name,
      item: item.item,
    })),
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
    />
  );
}

interface ArticleDataProps {
  headline: string;
  description: string;
  image: string;
  author: string;
  publishedAt: string;
  updatedAt?: string;
  keywords?: string[];
  url: string;
}

export function ArticleData({
  headline,
  description,
  image,
  author,
  publishedAt,
  updatedAt,
  keywords = [],
  url,
}: ArticleDataProps) {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline,
    description,
    image,
    author: {
      '@type': 'Person',
      name: author,
      url: 'https://mariazork.github.io',
      sameAs: [
        'https://github.com/MariaZork',
        'https://www.linkedin.com/in/maria-zorkaltseva/',
        'https://scholar.google.com/citations?user=kJHS8ygAAAAJ',
      ],
    },
    publisher: {
      '@type': 'Person',
      name: 'Maria Zorkaltseva',
      url: 'https://mariazork.github.io',
      logo: {
        '@type': 'ImageObject',
        url: 'https://mariazork.github.io/images/android-chrome-512x512.png',
      },
    },
    url,
    datePublished: publishedAt,
    ...(updatedAt && { dateModified: updatedAt }),
    ...(keywords.length > 0 && { keywords: keywords.join(', ') }),
    inLanguage: 'en',
    mainEntityOfPage: {
      '@type': 'WebPage',
      '@id': url,
    },
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
    />
  );
}

interface WebSiteDataProps {
  siteName?: string;
  url?: string;
  searchUrl?: string;
}

export function WebSiteData({
  siteName = 'Maria Zorkaltseva',
  url = 'https://mariazork.github.io',
  searchUrl = 'https://mariazork.github.io/blog/?q={search_term_string}',
}: WebSiteDataProps) {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: siteName,
    url,
    potentialAction: {
      '@type': 'SearchAction',
      target: {
        '@type': 'EntryPoint',
        urlTemplate: searchUrl,
      },
      'query-input': 'required name=search_term_string',
    },
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
    />
  );
}

interface PersonDataProps {
  name?: string;
  url?: string;
  jobTitle?: string;
  email?: string;
  address?: {
    city: string;
    country: string;
  };
  sameAs?: string[];
  knowsAbout?: string[];
}

export function PersonData({
  name = 'Maria Zorkaltseva',
  url = 'https://mariazork.github.io',
  jobTitle = 'Lead Machine Learning Engineer',
  email = 'maria.zorkaltseva@gmail.com',
  address = { city: 'Paris', country: 'FR' },
  sameAs = [
    'https://github.com/MariaZork',
    'https://www.linkedin.com/in/maria-zorkaltseva/',
    'https://twitter.com/MZorkaltseva',
    'https://scholar.google.com/citations?user=kJHS8ygAAAAJ',
    'https://medium.com/@maria.zorkaltseva',
  ],
  knowsAbout = [
    'Machine Learning',
    'Computer Vision',
    'NLP',
    'MLOps',
    'Deep Learning',
    'PyTorch',
    'Signal Processing',
    'Wi-Fi CSI Sensing',
    'Medical Imaging',
    'Generative AI',
    'LLMs',
  ],
}: PersonDataProps) {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'Person',
    name,
    url,
    jobTitle,
    email,
    address: {
      '@type': 'PostalAddress',
      addressLocality: address.city,
      addressCountry: address.country,
    },
    sameAs,
    knowsAbout,
    worksFor: {
      '@type': 'Organization',
      name: 'Freelance / Consulting',
    },
    alumniOf: [
      {
        '@type': 'EducationalOrganization',
        name: 'Tomsk State University',
      },
    ],
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
    />
  );
}

interface WebPageDataProps {
  name: string;
  description: string;
  url: string;
  image?: string;
  author?: string;
}

export function WebPageData({
  name,
  description,
  url,
  image = 'https://mariazork.github.io/images/android-chrome-512x512.png',
  author = 'Maria Zorkaltseva',
}: WebPageDataProps) {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'WebPage',
    name,
    description,
    url,
    image,
    author: {
      '@type': 'Person',
      name: author,
    },
    publisher: {
      '@type': 'Person',
      name: 'Maria Zorkaltseva',
      url: 'https://mariazork.github.io',
    },
    inLanguage: 'en',
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
    />
  );
}

interface FAQDataProps {
  questions: Array<{
    question: string;
    answer: string;
  }>;
}

export function FAQData({ questions }: FAQDataProps) {
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: questions.map((q) => ({
      '@type': 'Question',
      name: q.question,
      acceptedAnswer: {
        '@type': 'Answer',
        text: q.answer,
      },
    })),
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
    />
  );
}

export default {
  BreadcrumbData,
  ArticleData,
  WebSiteData,
  PersonData,
  WebPageData,
  FAQData,
};
