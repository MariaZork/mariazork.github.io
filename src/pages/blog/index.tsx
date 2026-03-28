import { GetStaticProps } from 'next';
import Layout from '@/components/Layout/Layout';
import PostCard from '@/components/Blog/PostCard';
import { Post } from '@/types/post';
import { getAllPosts } from '@/lib/posts';

interface BlogIndexProps {
  posts: Post[];
}

export default function BlogIndex({ posts }: BlogIndexProps) {
  return (
    <Layout
      title="Blog — Maria Zorkaltseva"
      description="Articles and tutorials on machine learning, deep learning, PyTorch, NLP, and AI in cybersecurity."
    >
      {/* Header */}
      <div className="border-b border-border bg-surface-alt">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
          <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-2">Writing</p>
          <h1 className="font-serif text-4xl md:text-5xl text-ink mb-4">Blog & Tutorials</h1>
          <p className="text-ink-muted max-w-xl leading-relaxed">
            Deep dives into machine learning theory, PyTorch experiments, computer vision, NLP, and AI in cybersecurity.
          </p>
        </div>
      </div>

      {/* Grid */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
        {posts.length === 0 ? (
          <div className="text-center py-24 text-ink-muted">
            <p className="font-serif text-2xl mb-2">No posts yet</p>
            <p className="text-sm">Check back soon.</p>
          </div>
        ) : (
          <>
            <p className="text-sm text-ink-muted mb-8">
              {posts.length} article{posts.length !== 1 ? 's' : ''}
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {posts.map((post) => (
                <PostCard key={post.slug} post={post} />
              ))}
            </div>
          </>
        )}
      </section>
    </Layout>
  );
}

export const getStaticProps: GetStaticProps = async () => {
  const posts = getAllPosts();
  return {
    props: {
      posts: JSON.parse(JSON.stringify(posts)),
    },
  };
};
