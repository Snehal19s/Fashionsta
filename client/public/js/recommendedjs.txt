// client/public/js/recommended.js
document.addEventListener('DOMContentLoaded', async () => {
  const container = document.getElementById('recommended-products');
  const loader    = document.getElementById('products-loader');
  const params    = new URLSearchParams(window.location.search);

  // Read the three recommendation arrays from the URL:
  const colors = params.get('colors')?.split(',').map(s=>s.trim()).filter(Boolean) || [];
  const styles = params.get('styles')?.split(',').map(s=>s.trim()).filter(Boolean) || [];
  const fits   = params.get('fits')?.split(',').map(s=>s.trim()).filter(Boolean)   || [];

  // If none provided, prompt the user:
  if (!colors.length && !styles.length && !fits.length) {
    loader.textContent = 'No recommendation parameters found.';
    return;
  }

  // Show loader
  loader.style.display = 'block';

  try {
    // Build the query string the same way your backend expects
    const qs = new URLSearchParams({
      colors: colors.join(','),
      styles: styles.join(','),
      fits:   fits.join(',')
    });

    // Fetch matching products
    const products = await makeApiRequest(`/products/recommendations?${qs.toString()}`);

    // Hide loader
    loader.style.display = 'none';

    if (!products.length) {
      container.innerHTML = '<p>No products match your style profile.</p>';
      return;
    }

    // Render each product card (reuse displayProductCard from products.js)
    products.forEach(p => displayProductCard(p, container));

  } catch (err) {
    loader.style.display = 'none';
    container.innerHTML = `<p class="error-message">Failed to load recommendations: ${err.message}</p>`;
  }
});
