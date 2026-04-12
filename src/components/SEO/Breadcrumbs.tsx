import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { BreadcrumbData } from './StructuredData';

interface BreadcrumbItem {
  name: string;
  href?: string;
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[];
  showSchema?: boolean;
}

export default function Breadcrumbs({ items, showSchema = true }: BreadcrumbsProps) {
  const router = useRouter();
  const siteUrl = 'https://mariazork.github.io';
  
  // Build structured data items
  const schemaItems = items.map((item, index) => ({
    name: item.name,
    item: item.href ? `${siteUrl}${item.href}` : `${siteUrl}${router.asPath}`,
  }));

  return (
    <>
      {showSchema && <BreadcrumbData items={schemaItems} />}
      <nav aria-label="Breadcrumb" className="py-3">
        <ol className="flex flex-wrap items-center gap-2 text-sm text-ink-muted">
          <li>
            <Link 
              href="/" 
              className="hover:text-primary transition-colors"
              aria-label="Home"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
            </Link>
          </li>
          {items.map((item, index) => (
            <React.Fragment key={index}>
              <li aria-hidden="true">
                <svg className="w-3 h-3 text-border" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </li>
              <li>
                {item.href && index < items.length - 1 ? (
                  <Link 
                    href={item.href} 
                    className="hover:text-primary transition-colors"
                  >
                    {item.name}
                  </Link>
                ) : (
                  <span className="text-ink" aria-current="page">
                    {item.name}
                  </span>
                )}
              </li>
            </React.Fragment>
          ))}
        </ol>
      </nav>
    </>
  );
}
