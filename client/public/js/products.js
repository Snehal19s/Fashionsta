// client/public/js/products.js

document.addEventListener('DOMContentLoaded', () => {
    const productGrid = document.getElementById('product-listing-grid'); // For product listing page
    const productDetailContainer = document.getElementById('product-detail-page-container'); // For product detail page
    const recommendationsGrid = document.getElementById('recommendations-grid'); // For homepage recommendations
    const reviewsContainer = document.getElementById('reviews-list');
    const addReviewForm = document.getElementById('add-review-form');
    const filterForm = document.getElementById('filter-form'); // Assuming you have a form for filters

    const urlParams = new URLSearchParams(window.location.search);
    const category = urlParams.get('category');
    const keyword = urlParams.get('keyword');
    const productId = urlParams.get('id'); // For product detail page


    // --- Product Listing Page & Filters ---
    if (productGrid) {
        loadProducts(); // Initial load

        if (filterForm) {
            filterForm.addEventListener('submit', (e) => {
                e.preventDefault();
                loadProducts(1, true); // Reset to page 1 and apply filters
            });
            // Add event listeners for individual filter changes if not using a submit button
            // E.g., filterForm.querySelectorAll('input, select').forEach(el => el.addEventListener('change', () => loadProducts(1, true)));
        }
    }

    async function loadProducts(page = 1, applyFilters = false) {
        showLoader('products-loader'); // Assume a loader with this ID
        productGrid.innerHTML = ''; // Clear previous products

        let query = `?pageNumber=${page}`;
        if (category) query += `&category=${encodeURIComponent(category)}`;
        if (keyword) query += `&keyword=${encodeURIComponent(keyword)}`;

        if (applyFilters && filterForm) {
            const formData = new FormData(filterForm);
            for (let [key, value] of formData.entries()) {
                if (value) { // Only add if filter has a value
                    query += `&${encodeURIComponent(key)}=${encodeURIComponent(value)}`;
                }
            }
        }
        // Or get values directly:
        // const minPrice = document.getElementById('minPrice')?.value;
        // if (minPrice) query += `&minPrice=${minPrice}`;

        try {
            const data = await makeApiRequest(`/products${query}`);
            if (data.products.length === 0) {
                productGrid.innerHTML = '<p class="empty-message">No products found matching your criteria.</p>';
            } else {
                data.products.forEach(product => displayProductCard(product, productGrid));
            }
            // displayPagination(data.page, data.pages, (newPage) => loadProducts(newPage, applyFilters));
        } catch (error) {
            productGrid.innerHTML = `<p class="error-message">Error loading products: ${error.message}</p>`;
        } finally {
            hideLoader('products-loader');
        }
    }

    // --- Product Detail Page ---
    if (productDetailContainer && productId) {
        loadProductDetails(productId);
        loadProductReviews(productId);

        if (addReviewForm) {
            addReviewForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                if (!getToken()) {
                    displayMessage('review-message', 'Please login to submit a review.', true);
                    return;
                }
                const rating = addReviewForm.rating.value;
                const comment = addReviewForm.comment.value;
                try {
                    await makeApiRequest(`/reviews/${productId}`, 'POST', { rating, comment }, true);
                    displayMessage('review-message', 'Review submitted successfully!', false);
                    addReviewForm.reset();
                    loadProductReviews(productId); // Refresh reviews
                    loadProductDetails(productId); // Refresh product rating
                } catch (error) {
                    displayMessage('review-message', error.message || 'Failed to submit review.', true);
                }
            });
        }
    }

    async function loadProductDetails(id) {
        showLoader('product-detail-loader');
        try {
            const product = await makeApiRequest(`/products/${id}`);
            displayProductDetails(product);
        } catch (error) {
            productDetailContainer.innerHTML = `<p class="error-message">Error loading product details: ${error.message}</p>`;
        } finally {
            hideLoader('product-detail-loader');
        }
    }

    function displayProductDetails(product) {
        // Assuming HTML structure for product details
        document.getElementById('product-main-image').src = `http://localhost:5001${product.images[0] || '/images/placeholder.png'}`;
        document.getElementById('product-name-detail').textContent = product.name;
        document.getElementById('product-brand-detail').textContent = product.brand || "Fashionsta"; // Add brand if available
        document.getElementById('product-price-detail').textContent = `$${product.price.toFixed(2)}`;
        document.getElementById('product-description-detail').textContent = product.description;
        document.getElementById('product-rating-detail').innerHTML = `Rating: ${product.ratings.toFixed(1)}/5 (${product.numOfReviews} reviews)`;

        const sizeOptionsContainer = document.getElementById('size-options');
        sizeOptionsContainer.innerHTML = '';
        if (product.sizes && product.sizes.length > 0) {
            product.sizes.forEach(size => {
                const button = document.createElement('button');
                button.textContent = size;
                button.dataset.size = size;
                button.onclick = () => {
                    // Handle size selection (e.g., add 'selected' class)
                    sizeOptionsContainer.querySelectorAll('button').forEach(btn => btn.classList.remove('selected'));
                    button.classList.add('selected');
                };
                sizeOptionsContainer.appendChild(button);
            });
        } else {
            sizeOptionsContainer.innerHTML = "<p>One size</p>"
        }


        const thumbnailContainer = document.getElementById('thumbnail-images');
        thumbnailContainer.innerHTML = '';
        product.images.forEach((imgUrl, index) => {
            const img = document.createElement('img');
            img.src = `http://localhost:5001${imgUrl}`;
            img.alt = `${product.name} thumbnail ${index + 1}`;
            if (index === 0) img.classList.add('active-thumb');
            img.onclick = () => {
                document.getElementById('product-main-image').src = imgUrl;
                thumbnailContainer.querySelectorAll('img').forEach(thumb => thumb.classList.remove('active-thumb'));
                img.classList.add('active-thumb');
            };
            thumbnailContainer.appendChild(img);
        });

        // Add to Cart Button on Detail Page
        const addToCartBtnDetail = document.getElementById('add-to-cart-btn-detail');
        if (addToCartBtnDetail) {
            addToCartBtnDetail.onclick = async () => {
                const selectedSizeEl = sizeOptionsContainer.querySelector('button.selected');
                const selectedSize = selectedSizeEl ? selectedSizeEl.dataset.size : (product.sizes && product.sizes.length > 0 ? null : "OneSize");
                const quantity = parseInt(document.getElementById('quantity-detail').value) || 1;

                if (product.sizes && product.sizes.length > 0 && !selectedSize) {
                    alert('Please select a size.');
                    return;
                }

                await handleAddToCart(product._id, quantity, selectedSize);
            };
        }

        // Add to Wishlist Button on Detail Page
        const addToWishlistBtnDetail = document.getElementById('add-to-wishlist-btn-detail');
        if (addToWishlistBtnDetail) {
            // Check if already in wishlist and update button state
            checkIfInWishlist(product._id, addToWishlistBtnDetail);
            addToWishlistBtnDetail.onclick = async () => {
                await handleToggleWishlist(product._id, addToWishlistBtnDetail);
            };
        }
    }

    // --- Reviews ---
    async function loadProductReviews(id) {
        if (!reviewsContainer) return;
        reviewsContainer.innerHTML = '<h4>Loading reviews...</h4>';
        try {
            const reviews = await makeApiRequest(`/reviews/${id}`);
            if (reviews.length === 0) {
                reviewsContainer.innerHTML = '<p>No reviews yet. Be the first to review!</p>';
            } else {
                reviewsContainer.innerHTML = '<h3>Customer Reviews</h3>';
                reviews.forEach(review => {
                    const reviewEl = document.createElement('div');
                    reviewEl.classList.add('review-item');
                    reviewEl.innerHTML = `
                        <div class="review-meta">
                            <strong>${review.userName}</strong> - ${new Date(review.createdAt).toLocaleDateString()}
                        </div>
                        <div class="review-rating">${'★'.repeat(review.rating)}${'☆'.repeat(5-review.rating)}</div>
                        <p class="review-comment">${review.comment}</p>
                    `;
                    reviewsContainer.appendChild(reviewEl);
                });
            }
        } catch (error) {
            reviewsContainer.innerHTML = `<p class="error-message">Could not load reviews: ${error.message}</p>`;
        }
    }

    // --- Recommendations ---
    if (recommendationsGrid) {
        loadRecommendedProducts();
    }
    async function loadRecommendedProducts() {
        if (!getToken()) { // Only show recommendations if logged in
            recommendationsGrid.parentElement.style.display = 'none'; // Hide the whole section
            return;
        }
        showLoader('recommendations-loader'); // Assuming a loader
        try {
            const products = await makeApiRequest('/products/recommendations', 'GET', null, true);
            recommendationsGrid.innerHTML = '';
            if (products.length === 0) {
                recommendationsGrid.parentElement.style.display = 'none';
            } else {
                recommendationsGrid.parentElement.style.display = 'block';
                products.forEach(product => displayProductCard(product, recommendationsGrid));
            }
        } catch (error) {
            console.error('Failed to load recommendations:', error);
            recommendationsGrid.parentElement.style.display = 'none';
        } finally {
            hideLoader('recommendations-loader');
        }
    }


    // --- Helper to display a single product card ---
    function displayProductCard(product, containerElement) {
        const card = document.createElement('div');
        card.className = 'product-card';
        card.innerHTML = `
            <a href="product-detail.html?id=${product._id}">
                <img src="http://localhost:5001${product.images[0] || '/images/placeholder.png'}" alt="${product.name}">
                <div class="product-info">
                    ${product.brand ? `<div class="brand-name">${product.brand}</div>` : ''}
                    <h3>${product.name}</h3>
                    <p class="price">₹${product.price.toFixed(2)}</p>
                    <p class="rating">
                        ${product.ratings > 0 ? `${'★'.repeat(Math.round(product.ratings))}${'☆'.repeat(5-Math.round(product.ratings))} (${product.numOfReviews})` : 'No reviews'}
                    </p>
                </div>
            </a>
            <div class="product-card-buttons">
                <button class="btn btn-primary add-to-cart-btn" data-product-id="${product._id}">Add to Cart</button>
                <button class="btn add-to-wishlist-btn" data-product-id="${product._id}">
                    <span class="wishlist-icon">♡</span> Wishlist
                </button>
            </div>
        `;
        containerElement.appendChild(card);

        // Add event listeners for buttons on the card
        const addToCartBtn = card.querySelector('.add-to-cart-btn');
        addToCartBtn.addEventListener('click', async (e) => {
            e.stopPropagation(); // Prevent link navigation
            // For product listing, size selection might be handled differently (e.g., modal or go to detail page)
            // For simplicity here, we assume "OneSize" or redirect/prompt for size
            if (product.sizes && product.sizes.length > 0 && product.category !== "Fashion Accessories") { // Accessories might not have sizes
                 alert("Please go to product page to select size.");
                 window.location.href = `product-detail.html?id=${product._id}`;
                 return;
            }
            await handleAddToCart(product._id, 1, product.sizes && product.sizes.length > 0 ? product.sizes[0] : "OneSize");
        });

        const addToWishlistBtn = card.querySelector('.add-to-wishlist-btn');
        checkIfInWishlist(product._id, addToWishlistBtn); // Check initial state
        addToWishlistBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            await handleToggleWishlist(product._id, addToWishlistBtn);
        });
    }

    // --- Shared Add to Cart/Wishlist Logic (can be moved to cart.js/wishlist.js if preferred) ---
    async function handleAddToCart(productId, quantity, size) {
        if (!getToken()) {
            alert('Please login to add items to your cart.');
            window.location.href = `login.html?redirect=${window.location.pathname}${window.location.search}`;
            return;
        }
        try {
            await makeApiRequest('/users/cart', 'POST', { productId, quantity, size }, true);
            alert('Product added to cart!'); // Replace with a nicer notification
            updateCartCount();
        } catch (error) {
            alert(`Failed to add to cart: ${error.message}`);
        }
    }

    let userWishlistCache = null; // Cache to avoid repeated API calls for wishlist status

    async function getUserWishlist() {
        if (!getToken()) return [];
        if (userWishlistCache) return userWishlistCache;
        try {
            userWishlistCache = await makeApiRequest('/users/wishlist', 'GET', null, true);
            return userWishlistCache;
        } catch (error) {
            console.error("Failed to fetch wishlist:", error);
            return [];
        }
    }
    // Call this when user logs in/out or wishlist is modified to clear cache
    window.clearUserWishlistCache = () => { userWishlistCache = null; };


    async function checkIfInWishlist(productId, buttonElement) {
        if (!getToken()) {
            buttonElement.querySelector('.wishlist-icon').textContent = '♡';
            buttonElement.classList.remove('in-wishlist');
            return false;
        }
        const wishlist = await getUserWishlist();
        const isInWishlist = wishlist.some(item => item._id === productId);
        if (isInWishlist) {
            buttonElement.querySelector('.wishlist-icon').textContent = '❤️'; // Filled heart
            buttonElement.classList.add('in-wishlist');
        } else {
            buttonElement.querySelector('.wishlist-icon').textContent = '♡'; // Empty heart
            buttonElement.classList.remove('in-wishlist');
        }
        return isInWishlist;
    }

    async function handleToggleWishlist(productId, buttonElement) {
        if (!getToken()) {
            alert('Please login to manage your wishlist.');
            window.location.href = `login.html?redirect=${window.location.pathname}${window.location.search}`;
            return;
        }
        try {
            const isInWishlist = buttonElement.classList.contains('in-wishlist');
            if (isInWishlist) {
                await makeApiRequest(`/users/wishlist/${productId}`, 'DELETE', null, true);
                buttonElement.querySelector('.wishlist-icon').textContent = '♡';
                buttonElement.classList.remove('in-wishlist');
                alert('Removed from wishlist!');
            } else {
                await makeApiRequest('/users/wishlist', 'POST', { productId }, true);
                buttonElement.querySelector('.wishlist-icon').textContent = '❤️';
                buttonElement.classList.add('in-wishlist');
                alert('Added to wishlist!');
            }
            clearUserWishlistCache(); // Invalidate cache
            updateWishlistCount();
        } catch (error) {
            alert(`Wishlist operation failed: ${error.message}`);
        }
    }

});
// client/public/js/products.js

// ... (existing code) ...

// Function to display pagination controls
function displayPagination(currentPage, totalPages, onPageClickCallback) {
    const paginationContainer = document.getElementById('pagination-controls');
    if (!paginationContainer) return;
    paginationContainer.innerHTML = ''; // Clear old controls

    if (totalPages <= 1) return; // No pagination needed for 1 or less pages

    const ul = document.createElement('ul');
    ul.className = 'pagination'; // Add a class for styling if needed

    // Previous Button
    if (currentPage > 1) {
        const prevLi = document.createElement('li');
        const prevA = document.createElement('a');
        prevA.href = '#';
        prevA.textContent = '« Previous';
        prevA.addEventListener('click', (e) => {
            e.preventDefault();
            onPageClickCallback(currentPage - 1);
        });
        prevLi.appendChild(prevA);
        ul.appendChild(prevLi);
    }

    // Page Number Buttons (simplified: show a few pages around current)
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, currentPage + 2);

    if (startPage > 1) {
        const firstLi = document.createElement('li');
        const firstA = document.createElement('a');
        firstA.href = '#';
        firstA.textContent = '1';
        firstA.addEventListener('click', (e) => { e.preventDefault(); onPageClickCallback(1); });
        firstLi.appendChild(firstA);
        ul.appendChild(firstLi);
        if (startPage > 2) ul.appendChild(document.createTextNode('... '));
    }

    for (let i = startPage; i <= endPage; i++) {
        const pageLi = document.createElement('li');
        const pageA = document.createElement('a');
        pageA.href = '#';
        pageA.textContent = i;
        if (i === currentPage) {
            pageA.classList.add('active'); // Style the active page
        }
        pageA.addEventListener('click', (e) => {
            e.preventDefault();
            if (i !== currentPage) {
                onPageClickCallback(i);
            }
        });
        pageLi.appendChild(pageA);
        ul.appendChild(pageLi);
    }

     if (endPage < totalPages) {
        if (endPage < totalPages - 1) ul.appendChild(document.createTextNode(' ...'));
        const lastLi = document.createElement('li');
        const lastA = document.createElement('a');
        lastA.href = '#';
        lastA.textContent = totalPages;
        lastA.addEventListener('click', (e) => { e.preventDefault(); onPageClickCallback(totalPages); });
        lastLi.appendChild(lastA);
        ul.appendChild(lastLi);
    }


    // Next Button
    if (currentPage < totalPages) {
        const nextLi = document.createElement('li');
        const nextA = document.createElement('a');
        nextA.href = '#';
        nextA.textContent = 'Next »';
        nextA.addEventListener('click', (e) => {
            e.preventDefault();
            onPageClickCallback(currentPage + 1);
        });
        nextLi.appendChild(nextA);
        ul.appendChild(nextLi);
    }
    paginationContainer.appendChild(ul);
}


async function loadProducts(page = 1, applyFilters = false) {
    showLoader('products-loader');
    productGrid.innerHTML = '';

    let query = `?pageNumber=${page}`;
    // ... (rest of your query building logic) ...
    const urlParams = new URLSearchParams(window.location.search); // Get current URL params
    const currentCategory = urlParams.get('category') || (filterForm.category ? filterForm.category.value : '');
    const currentKeyword = urlParams.get('keyword') || (filterForm.keyword ? filterForm.keyword.value : '');

    if (currentCategory) query += `&category=${encodeURIComponent(currentCategory)}`;
    if (currentKeyword) query += `&keyword=${encodeURIComponent(currentKeyword)}`;


    if (applyFilters && filterForm) {
        const formData = new FormData(filterForm);
        for (let [key, value] of formData.entries()) {
            if (value && key !== 'category' && key !== 'keyword') { // Don't re-add category/keyword if already in base query
                query += `&${encodeURIComponent(key)}=${encodeURIComponent(value)}`;
            }
        }
    }

    try {
        const data = await makeApiRequest(`/products${query}`);
        if (data.products.length === 0) {
            productGrid.innerHTML = '<p class="empty-message">No products found matching your criteria.</p>';
        } else {
            data.products.forEach(product => displayProductCard(product, productGrid));
        }
        // Call displayPagination here
        displayPagination(data.page, data.pages, (newPage) => {
            // When a pagination link is clicked, update URL and reload products
            const newUrlParams = new URLSearchParams(window.location.search);
            newUrlParams.set('pageNumber', newPage);
            // window.history.pushState({path: `${window.location.pathname}?${newUrlParams.toString()}`}, '', `${window.location.pathname}?${newUrlParams.toString()}`);
            // Simpler for now: just reload with the new page
            loadProducts(newPage, true); // Pass true to re-apply current filters
        });
    } catch (error) {
        productGrid.innerHTML = `<p class="error-message">Error loading products: ${error.message}</p>`;
    } finally {
        hideLoader('products-loader');
    }
}
// ... (rest of products.js)