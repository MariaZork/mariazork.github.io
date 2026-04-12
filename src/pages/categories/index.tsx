import React from 'react';
import Link from 'next/link';
import Layout from '@/components/Layout/Layout';
import Breadcrumbs from '@/components/SEO/Breadcrumbs';
import { getAllPosts } from '@/lib/posts';

interface CategoryInfo {
  name: string;
  slug: string;
  count: number;
}

interface CategoriesIndexProps {
  categories: CategoryInfo[];
}

const breadcrumbItems = [
  { name: 'Categories', href: '/categories/' },
];

export default function CategoriesIndex({ categories }: CategoriesIndexProps) {
  return (
    <Layout
      title="Categories — Maria Zorkaltseva"
      description="Browse all articles organized by technical topic."
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <Breadcrumbs items={breadcrumbItems} />
      </div>
      
      <div className="border-b border-border bg-surface-alt">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
          <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-2">Browse</p>
          <h1 className="font-serif text-4xl md:text-5xl text-ink mb-4">Categories</h1>
          <p className="text-ink-muted max-w-xl">
            Articles organized by technical topic and research area.
          </p>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-14">
        {categories.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {categories.map((cat) => (
              <Link
                key={cat.slug}
                href={`/categories/${cat.slug}`}
                className="group bg-white p-5 rounded-2xl border border-border shadow-card hover:shadow-card-hover hover:border-primary/30 transition-all flex items-center justify-between"
              >
                <div className="flex items-center gap-4">
                  <div className="w-11 h-11 bg-primary-soft text-primary rounded-xl flex items-center justify-center group-hover:bg-primary group-hover:text-white transition-colors flex-shrink-0">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                    </svg>
                  </div>
                  <div>
                    <span className="text-base font-semibold text-ink group-hover:text-primary transition-colors block">
                      {cat.name}
                    </span>
                    <span className="text-xs text-ink-muted">
                      {cat.count} {cat.count === 1 ? 'article' : 'articles'}
                    </span>
                  </div>
                </div>
                <svg className="w-4 h-4 text-border group-hover:text-primary group-hover:translate-x-0.5 transition-all" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            ))}
          </div>
        ) : (
          <div className="text-center py-24 bg-white rounded-2xl border border-dashed border-border">
            <p className="font-serif text-xl text-ink-muted">No categories found yet.</p>
          </div>
        )}
      </div>
    </Layout>
  );
}

export const getStaticProps = async () => {
  try {
    const posts = getAllPosts();
    const categoryCounts: Record<string, { name: string; count: number }> = {};

    posts.forEach(post => {
      if (post.categories) {
        const cats = Array.isArray(post.categories) ? post.categories : [post.categories];
        cats.forEach(cat => {
          if (typeof cat === 'string' && cat.trim()) {
            const lower = cat.trim().toLowerCase();
            categoryCounts[lower] = categoryCounts[lower]
              ? { ...categoryCounts[lower], count: categoryCounts[lower].count + 1 }
              : { name: cat.trim(), count: 1 };
          }
        });
      }
    });

    const categories = Object.keys(categoryCounts).map(key => ({
      name: categoryCounts[key].name,
      slug: key.replace(/\s+/g, '-'),
      count: categoryCounts[key].count,
    })).sort((a, b) => b.count - a.count);

    return { props: { categories: JSON.parse(JSON.stringify(categories)) } };
  } catch {
    return { props: { categories: [] } };
  }
};
