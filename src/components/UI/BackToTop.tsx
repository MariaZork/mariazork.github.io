import React, { useEffect, useState } from 'react';

const BackToTop: React.FC = () => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setVisible(window.scrollY > 300);
    };

    onScroll();
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const handleClick = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (!visible) return null;

  return (
    <button
      type="button"
      onClick={handleClick}
      className="fixed bottom-6 right-6 z-40 rounded-full bg-primary text-white shadow-lg px-4 py-3 text-sm font-semibold hover:bg-primary-dark transition"
      aria-label="Back to top"
    >
      ↑ Top
    </button>
  );
};

export default BackToTop;
