import { useCallback } from 'react';

interface SocialShareProps {
  title: string;
  slug: string;
}

const SocialShare: React.FC<SocialShareProps> = ({ title, slug }) => {
  const baseUrl =
    process.env.NEXT_PUBLIC_SITE_URL || 'https://mariazork.github.io';
  const url = `${baseUrl}/blog/${slug}`;

  const twitterHref = `https://twitter.com/intent/tweet?url=${encodeURIComponent(
    url,
  )}&text=${encodeURIComponent(title)}`;

  const linkedinHref = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(
    url,
  )}`;

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(url);
      alert('Link copied to clipboard ✨');
    } catch {
      alert('Failed to copy link');
    }
  }, [url]);

  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="text-sm font-semibold text-gray-600">
        Share this article:
      </span>

      <a
        href={twitterHref}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm px-3 py-1 rounded-full bg-[#1DA1F2]/10 text-[#1DA1F2] hover:bg-[#1DA1F2]/20 transition"
      >
        Twitter
      </a>

      <a
        href={linkedinHref}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm px-3 py-1 rounded-full bg-[#0A66C2]/10 text-[#0A66C2] hover:bg-[#0A66C2]/20 transition"
      >
        LinkedIn
      </a>

      <button
        type="button"
        onClick={handleCopy}
        className="text-sm px-3 py-1 rounded-full bg-gray-100 text-gray-700 hover:bg-gray-200 transition"
      >
        Copy link
      </button>
    </div>
  );
};

export default SocialShare;
