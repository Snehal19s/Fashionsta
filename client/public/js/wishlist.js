// client/public/js/wishlist.js

document.addEventListener('DOMContentLoaded', () => {
    const wishlistItemsContainer = document.getElementById('wishlist-items-container');
    const emptyWishlistMessage = document.getElementById('empty-wishlist-message');


    if (wishlistItemsContainer) { // Only run on wishlist page
        loadWishlist();
    }

    async function loadWishlist() {
        if (!getToken()) {
            wishlistItemsContainer.innerHTML = '';
            if(emptyWishlistMessage) emptyWishlistMessage.style.display = 'block';
            // Optionally redirect to login or show a "please login" message
            return;
        }

        showLoader('wishlist-loader'); // Assume a loader
        try {
            const wishlist = await makeApiRequest('/users/wishlist', 'GET', null, true);
            wishlistItemsContainer.innerHTML = ''; // Clear previous items
            if (wishlist.length === 0) {
                if(emptyWishlistMessage) emptyWishlistMessage.style.display = 'block';
            } else {
                if(emptyWishlistMessage) emptyWishlistMessage.style.display = 'none';
                wishlist.forEach(product => displayWishlistItem(product));
            }
            updateWishlistCount(); // Update header count
            if(window.clearUserWishlistCache) window.clearUserWishlistCache(); // Clear cache used in products.js
        } catch (error) {
            wishlistItemsContainer.innerHTML = `<p class="error-message">Failed to load wishlist: ${error.message}</p>`;
            if(emptyWishlistMessage) emptyWishlistMessage.style.display = 'none';
        } finally {
            hideLoader('wishlist-loader');
        }
    }

    function displayWishlistItem(product) {
        const itemElement = document.createElement('div');
        itemElement.className = 'wishlist-item';
        itemElement.dataset.productId = product._id;
        itemElement.innerHTML = `
            <div class="wishlist-item-image">
                <a href="product-detail.html?id=${product._id}">
                    <img src="${product.images[0].startsWith('/uploads/') ? 'http://localhost:5001' + product.images[0] : 'http://localhost:5001/uploads/' + product.images[0]}" alt="${product.name}">
                </a>
            </div>
            <div class="wishlist-item-details">
                <h3><a href="product-detail.html?id=${product._id}">${product.name}</a></h3>
                <p class="price">₹${product.price.toFixed(2)}</p>
                <!-- Add rating or other info if desired -->
            </div>
            <div class="wishlist-item-actions">
                <button class="btn btn-sm btn-primary move-to-cart-btn" data-product-id="${product._id}">Move to Cart</button>
                <button class="remove-from-wishlist-btn" data-product-id="${product._id}">Remove</button>
            </div>
        `;
        wishlistItemsContainer.appendChild(itemElement);

        // Event listeners
        itemElement.querySelector('.remove-from-wishlist-btn').addEventListener('click', async (e) => {
            const productId = e.target.dataset.productId;
            if (confirm('Are you sure you want to remove this item from your wishlist?')) {
                await removeFromWishlist(productId);
            }
        });

        itemElement.querySelector('.move-to-cart-btn').addEventListener('click', async (e) => {
            const productId = e.target.dataset.productId;
            if (product.sizes && product.sizes.length > 0 && product.category !== "Fashion Accessories") {
                alert("Please select size on product page before adding to cart from wishlist.");
                window.location.href = `product-detail.html?id=${productId}`;
                return;
            }
            const sizeToAddToCart = (product.sizes && product.sizes.length > 0) ? product.sizes[0] : "OneSize";

            await globalAddToCart(productId, 1, sizeToAddToCart);
            await removeFromWishlist(productId);
        });
    }

    async function removeFromWishlist(productId) {
        try {
            await makeApiRequest(`/users/wishlist/${productId}`, 'DELETE', null, true);
            loadWishlist(); // Reload wishlist
        } catch (error) {
            alert(`Failed to remove item from wishlist: ${error.message}`);
        }
    }
});

// Global functions for adding/removing (can be called from product.js)
async function globalAddToWishlist(productId) {
    if (!getToken()) {
        alert('Please login to add items to your wishlist.');
        window.location.href = `login.html?redirect=${window.location.pathname}${window.location.search}`;
        return false;
    }
    try {
        await makeApiRequest('/users/wishlist', 'POST', { productId }, true);
        alert('Added to wishlist!');
        updateWishlistCount();
        if(window.clearUserWishlistCache) window.clearUserWishlistCache();
        return true;
    } catch (error) {
        alert(`Failed to add to wishlist: ${error.message}`);
        return false;
    }
}

async function globalRemoveFromWishlist(productId) {
     if (!getToken()) return false; // Should not happen if button is shown correctly
    try {
        await makeApiRequest(`/users/wishlist/${productId}`, 'DELETE', null, true);
        alert('Removed from wishlist!');
        updateWishlistCount();
        if(window.clearUserWishlistCache) window.clearUserWishlistCache();
        return true;
    } catch (error) {
        alert(`Failed to remove from wishlist: ${error.message}`);
        return false;
    }
}