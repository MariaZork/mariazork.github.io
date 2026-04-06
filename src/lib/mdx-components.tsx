import CodeCell, { CodeCellProvider } from '@/components/Code/CodeCell';
import BlogImage from '@/components/Blog/BlogImage';

export const mdxComponents = {
  CodeCell,
  BlogImage,
  img: ({ src, alt }: { src?: string; alt?: string }) => {
    if (!src) return null;

    const isBadge = src.includes('badge') || 
                    src.includes('shields.io') || 
                    src.includes('colab.research.google.com');
    
    if (isBadge) {
        return (
        <span className="block text-right my-2">
          <img 
            src={src} 
            alt={alt ?? ''} 
            className="inline-block"
            style={{ height: '28px', width: 'auto' }}
          />
        </span>
      );
    }
    
    return <BlogImage src={src} alt={alt ?? ''} zoomable />;
  },
};

export { CodeCellProvider };