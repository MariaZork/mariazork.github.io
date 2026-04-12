import React from 'react';
import Link from 'next/link';
import Layout from '@/components/Layout/Layout';
import Breadcrumbs from '@/components/SEO/Breadcrumbs';
import { getAllPosts } from '@/lib/posts';

interface TagInfo {
  name: string;
  slug: string;
  count: number;
}

interface TagsIndexProps {
  tags: TagInfo[];
}

const breadcrumbItems = [
  { name: 'Tags', href: '/tags/' },
];

export default function TagsIndex({ tags }: TagsIndexProps) {
  return (
    <Layout
      title="Tags — Maria Zorkaltseva"
      description="Find articles by specific keywords and technologies."
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <Breadcrumbs items={breadcrumbItems} />
      </div>
      
      <div className="border-b border-border bg-surface-alt">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
          <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-2">Browse</p>
          <h1 className="font-serif text-4xl md:text-5xl text-ink mb-4">Tags</h1>
          <p className="text-ink-muted max-w-xl">
            Find articles by specific keywords, technologies, and methods.
          </p>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-14">
        {tags.length > 0 ? (
          <div className="flex flex-wrap gap-3">
            {tags.map((tag) => (
              <Link
                key={tag.slug}
                href={`/tags/${tag.slug}`}
                className="group inline-flex items-center gap-2.5 bg-white px-4 py-2.5 rounded-xl border border-border shadow-soft hover:shadow-card hover:border-primary/30 transition-all tag-pill"
              >
                <span className="text-xs text-ink-muted group-hover:text-primary transition-colors">#</span>
                <span className="text-sm font-semibold text-ink group-hover:text-primary transition-colors">
                  {tag.name}
                </span>
                <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-surface-alt text-ink-muted group-hover:bg-primary-soft group-hover:text-primary transition-colors">
                  {tag.count}
                </span>
              </Link>
            ))}
          </div>
        ) : (
          <div className="text-center py-24 bg-white rounded-2xl border border-dashed border-border">
            <p className="font-serif text-xl text-ink-muted">No tags found yet.</p>
          </div>
        )}
      </div>
    </Layout>
  );
}

export const getStaticProps = async () => {
  try {
    const posts = getAllPosts();
    const tagCounts: Record<string, { name: string; count: number }> = {};

    posts.forEach(post => {
      if (post.tags) {
        const postTags = Array.isArray(post.tags) ? post.tags : [post.tags];
        postTags.forEach(tag => {
          if (typeof tag === 'string' && tag.trim()) {
            const lower = tag.trim().toLowerCase();
            tagCounts[lower] = tagCounts[lower]
              ? { ...tagCounts[lower], count: tagCounts[lower].count + 1 }
              : { name: tag.trim(), count: 1 };
          }
        });
      }
    });

    const tags = Object.keys(tagCounts).map(key => ({
      name: tagCounts[key].name,
      slug: key.replace(/\s+/g, '-'),
      count: tagCounts[key].count,
    })).sort((a, b) => b.count - a.count);

    return { props: { tags: JSON.parse(JSON.stringify(tags)) } };
  } catch {
    return { props: { tags: [] } };
  }
};
