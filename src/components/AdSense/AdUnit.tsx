import { useEffect } from 'react';

interface AdUnitProps {
  slot: 'display' | 'in-article' | 'sidebar';
}

const SLOT_IDS = {
  display: '9876543210',
  'in-article': '1234567890',
  sidebar: '1122334455',
};

export default function AdUnit({ slot }: AdUnitProps) {
  useEffect(() => {
    try {
      if (typeof window !== 'undefined') {
        ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
      }
    } catch (err) {
      console.error('AdSense error:', err);
    }
  }, []);

  return (
    <div className="adsense-container text-center my-8 p-4 bg-gray-50 border border-gray-200 rounded-lg">
      <ins
        className="adsbygoogle"
        style={{ display: 'block' }}
        data-ad-client="ca-pub-5520389121387465"
        data-ad-slot={SLOT_IDS[slot]}
        data-ad-format="auto"
        data-full-width-responsive="true"
      />
    </div>
  );
}