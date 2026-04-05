import { GetStaticProps } from 'next';
import Link from 'next/link';
import Layout from '@/components/Layout/Layout';
import PostCard from '@/components/Blog/PostCard';
import { Post } from '@/types/post';
import { getAllPosts } from '@/lib/posts';

interface HomeProps {
  posts: Post[];
}

const expertise = [
  { icon: '🧠', label: 'Applied Machine Learning' },
  { icon: '👁️', label: 'Computer Vision' },
  { icon: '💬', label: 'NLP' },
  { icon: '🤖', label: 'Generative AI & LLMs' },
  { icon: '⚙️', label: 'MLOps' },
  { icon: '📊', label: 'Edge AI & TinyML' },
];

export default function Home({ posts }: HomeProps) {
  return (
    <Layout>
      {/* ── Hero ── */}
      <section className="relative overflow-hidden border-b border-border">
        {/* Background texture */}
        <div
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage: `radial-gradient(circle at 20% 50%, rgba(26,188,156,0.12) 0%, transparent 50%),
                              radial-gradient(circle at 80% 20%, rgba(107,92,231,0.10) 0%, transparent 50%)`,
          }}
        />
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 py-20 md:py-28">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary-soft border border-primary/20 text-primary text-xs font-semibold mb-6">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              ML Engineer · Paris, France
            </div>
            <h1 className="font-serif text-5xl md:text-6xl text-ink leading-tight mb-5">
              Maria<br />
              <span className="text-primary">Zorkaltseva</span>
            </h1>
            <p className="text-lg text-ink-muted leading-relaxed mb-8 max-w-xl">
              Computer Vision and NLP specialist writing about machine learning theory,
              deep learning experiments, AI in cybersecurity and healthcare.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/blog"
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-ink text-white rounded-xl text-sm font-semibold hover:bg-ink/80 transition-all"
              >
                Read the Blog
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </Link>
              <Link
                href="/about"
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-white border border-border text-ink rounded-xl text-sm font-semibold hover:border-primary/40 hover:text-primary transition-all"
              >
                About me
              </Link>
            </div>
          </div>

          {/* Expertise pills */}
          <div className="flex flex-wrap gap-2 mt-12">
            {expertise.map(({ icon, label }) => (
              <span
                key={label}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-border rounded-full text-sm text-ink-muted shadow-soft"
              >
                <span>{icon}</span>
                {label}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Latest Articles ── */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 py-16">
        <div className="flex items-end justify-between mb-10">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-1">
              Recent writing
            </p>
            <h2 className="font-serif text-3xl text-ink">Latest Articles</h2>
          </div>
          <Link
            href="/blog"
            className="hidden sm:inline-flex items-center gap-1.5 text-sm font-medium text-ink-muted hover:text-primary transition-colors"
          >
            All posts
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </Link>
        </div>

        {posts.length === 0 ? (
          <div className="text-center py-16 text-ink-muted">
            <p className="font-serif text-2xl mb-2">No posts yet</p>
            <p className="text-sm">Check back soon.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {posts.map((post) => (
              <PostCard key={post.slug} post={post} />
            ))}
          </div>
        )}

        <div className="text-center mt-10 sm:hidden">
          <Link
            href="/blog"
            className="inline-flex items-center gap-2 px-6 py-2.5 border border-border rounded-xl text-sm font-medium text-ink hover:border-primary/40 hover:text-primary transition-all"
          >
            View all posts
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </Link>
        </div>
      </section>

      {/* ── Topics strip ── */}
      <section className="border-t border-border bg-surface-alt">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
          <p className="text-xs font-semibold uppercase tracking-widest text-ink-muted mb-5 text-center">
            Browse by topic
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              { label: 'Machine Learning', href: '/categories/machine-learning' },
              { label: 'Deep Learning', href: '/categories/deep-learning' },
              { label: 'PyTorch', href: '/tags/pytorch' },
              { label: 'Cybersecurity', href: '/categories/cybersecurity' },
              { label: 'Computer Vision', href: '/tags/images-classification' },
              { label: 'NLP', href: '/tags/machine-learning' },
              { label: 'Time Series', href: '/tags/time-series' },
              { label: 'Graph NN', href: '/tags/graph-neural-networks' },
            ].map(({ label, href }) => (
              <Link
                key={label}
                href={href}
                className="px-4 py-2 bg-white border border-border rounded-xl text-sm text-ink-muted hover:border-primary/40 hover:text-primary shadow-soft tag-pill transition-all"
              >
                {label}
              </Link>
            ))}
          </div>
        </div>
      </section>
    </Layout>
  );
}

export const getStaticProps: GetStaticProps = async () => {
  const allPosts = getAllPosts();
  const posts = allPosts.slice(0, 6);
  return {
    props: {
      posts: JSON.parse(JSON.stringify(posts)),
    },
  };
};
