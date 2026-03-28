import { GetStaticPaths, GetStaticProps } from 'next';
import Link from 'next/link';
import Layout from '@/components/Layout/Layout';
import PostCard from '@/components/Blog/PostCard';
import { getAllPosts } from '@/lib/posts';
import { Post } from '@/types/post';

interface TagPageProps {
  tag: string;
  posts: Post[];
}

export default function TagPage({ tag, posts }: TagPageProps) {
  return (
    <Layout
      title={`#${tag} — Maria Zorkaltseva`}
      description={`Articles tagged with "${tag}" by Maria Zorkaltseva.`}
    >
      <div className="border-b border-border bg-surface-alt">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-secondary mb-3">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
            </svg>
            Tag
          </div>
          <h1 className="font-serif text-4xl md:text-5xl text-ink mb-3">#{tag}</h1>
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
            <p className="font-serif text-xl text-ink-muted">No posts with this tag yet.</p>
          </div>
        )}

        <div className="mt-12">
          <Link
            href="/tags"
            className="inline-flex items-center gap-2 text-sm text-ink-muted hover:text-primary transition-colors group"
          >
            <svg className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            All tags
          </Link>
        </div>
      </div>
    </Layout>
  );
}

export const getStaticPaths: GetStaticPaths = async () => {
  const posts = getAllPosts();
  const tagSet = new Set<string>();
  posts.forEach(post => {
    const tags = Array.isArray(post.tags) ? post.tags : [post.tags];
    tags.forEach((t: string) => t && tagSet.add(t.trim()));
  });
  return {
    paths: Array.from(tagSet).map(t => ({
      params: { slug: t.toLowerCase().replace(/\s+/g, '-') },
    })),
    fallback: false,
  };
};

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const slug = params?.slug as string;
  const allPosts = getAllPosts();
  const filteredPosts = allPosts.filter(post => {
    const tags = Array.isArray(post.tags) ? post.tags : [post.tags];
    return tags.some((t: string) => t && t.toLowerCase().replace(/\s+/g, '-') === slug);
  });
  return {
    props: {
      tag: slug,
      posts: JSON.parse(JSON.stringify(filteredPosts)),
    },
  };
};
