import { GetStaticPaths, GetStaticProps } from 'next';
import Link from 'next/link';
import Layout from '@/components/Layout/Layout';
import PostCard from '@/components/Blog/PostCard';
import { getAllPosts } from '@/lib/posts';
import { Post } from '@/types/post';

interface CategoryPageProps {
  categoryName: string;
  posts: Post[];
}

export default function CategoryPage({ categoryName, posts }: CategoryPageProps) {
  const displayTitle = categoryName
    .split('-')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');

  return (
    <Layout
      title={`${displayTitle} — Maria Zorkaltseva`}
      description={`Articles about ${displayTitle} by Maria Zorkaltseva.`}
    >
      <div className="border-b border-border bg-surface-alt">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-primary mb-3">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
            </svg>
            Category
          </div>
          <h1 className="font-serif text-4xl md:text-5xl text-ink mb-3">{displayTitle}</h1>
          <p className="text-ink-muted">
            {posts.length} {posts.length === 1 ? 'article' : 'articles'}
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
        {posts.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {posts.map(post => <PostCard key={post.slug} post={post} />)}
          </div>
        ) : (
          <div className="text-center py-24 bg-white rounded-2xl border border-dashed border-border">
            <p className="font-serif text-xl text-ink-muted">No posts in this category yet.</p>
          </div>
        )}

        <div className="mt-12">
          <Link
            href="/categories"
            className="inline-flex items-center gap-2 text-sm text-ink-muted hover:text-primary transition-colors group"
          >
            <svg className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            All categories
          </Link>
        </div>
      </div>
    </Layout>
  );
}

export const getStaticPaths: GetStaticPaths = async () => {
  const posts = getAllPosts();
  const categorySet = new Set<string>();
  posts.forEach(post => {
    const cats = Array.isArray(post.categories) ? post.categories : [post.categories];
    cats.forEach((c: string) => categorySet.add(c.trim()));
  });
  return {
    paths: Array.from(categorySet).map(c => ({
      params: { slug: c.toLowerCase().replace(/\s+/g, '-') },
    })),
    fallback: false,
  };
};

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const slug = params?.slug as string;
  const allPosts = getAllPosts();
  const filteredPosts = allPosts.filter(post => {
    const cats = Array.isArray(post.categories) ? post.categories : [post.categories];
    return cats.some((c: string) => c.toLowerCase().replace(/\s+/g, '-') === slug);
  });
  return {
    props: {
      categoryName: slug,
      posts: JSON.parse(JSON.stringify(filteredPosts)),
    },
  };
};
