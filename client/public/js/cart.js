// client/public/js/cart.js

document.addEventListener('DOMContentLoaded', () => {
    const cartItemsContainer = document.getElementById('cart-items-container');
    const cartSummaryContainer = document.getElementById('cart-summary'); // For total price, etc.
    const emptyCartMessage = document.getElementById('empty-cart-message');

    if (cartItemsContainer) { // Only run on cart page
        loadCart();
    }

    async function loadCart() {
        if (!getToken()) {
            cartItemsContainer.innerHTML = '';
            if(emptyCartMessage) emptyCartMessage.style.display = 'block';
            if(cartSummaryContainer) cartSummaryContainer.style.display = 'none';
            // Optionally redirect to login or show a "please login" message
            return;
        }

        showLoader('cart-loader'); // Assume a loader
        try {
            const cart = await makeApiRequest('/users/cart', 'GET', null, true);
            cartItemsContainer.innerHTML = ''; // Clear previous items
            if (cart.length === 0) {
                if(emptyCartMessage) emptyCartMessage.style.display = 'block';
                if(cartSummaryContainer) cartSummaryContainer.style.display = 'none';
            } else {
                if(emptyCartMessage) emptyCartMessage.style.display = 'none';
                if(cartSummaryContainer) cartSummaryContainer.style.display = 'block';
                cart.forEach(item => displayCartItem(item));
                updateCartSummary(cart);
            }
            updateCartCount(); // Update header count
        } catch (error) {
            cartItemsContainer.innerHTML = `<p class="error-message">Failed to load cart: ${error.message}</p>`;
            if(emptyCartMessage) emptyCartMessage.style.display = 'none';
            if(cartSummaryContainer) cartSummaryContainer.style.display = 'none';
        } finally {
            hideLoader('cart-loader');
        }
    }

    function displayCartItem(item) {
        if (!item.product) {
            console.warn("Cart item missing product data:", item);
            // Optionally remove this item from cart automatically or show an error
            return;
        }
        const itemElement = document.createElement('div');
        itemElement.className = 'cart-item';
        itemElement.dataset.itemId = item._id; // The _id of the cart subdocument
        itemElement.innerHTML = `
            <div class="cart-item-image">
                <a href="product-detail.html?id=${item.product._id}">
                    <img src="${item.product.images[0].startsWith('/uploads/')  ? 'http://localhost:5001' + item.product.images[0]    : 'http://localhost:5001/uploads/' + item.product.images[0]}" alt="${item.product.name}">
                </a>
            </div>
            <div class="cart-item-details">
                <h3><a href="product-detail.html?id=${item.product._id}">${item.product.name}</a></h3>
                <p class="price">₹${item.product.price.toFixed(2)}</p>
                <p class="size">Size: ${item.size || 'N/A'}</p>
                <div class="cart-item-quantity">
                    <label for="quantity-${item._id}">Qty:</label>
                    <input type="number" id="quantity-${item._id}" value="${item.quantity}" min="1" data-item-id="${item._id}" class="item-quantity-input">
                </div>
            </div>
            <div class="cart-item-subtotal">
                <p>Subtotal: ₹${(item.product.price * item.quantity).toFixed(2)}</p>
            </div>
            <div class="cart-item-actions">
                <button class="remove-from-cart-btn" data-item-id="${item._id}">Remove</button>
            </div>
        `;
        cartItemsContainer.appendChild(itemElement);

        // Event listeners for quantity change and remove
        itemElement.querySelector('.item-quantity-input').addEventListener('change', async (e) => {
            const newQuantity = parseInt(e.target.value);
            const cartItemId = e.target.dataset.itemId;
            if (newQuantity >= 1) {
                await updateCartItemQuantity(cartItemId, newQuantity);
            } else {
                // Reset to 1 if invalid value, or handle removal
                e.target.value = 1;
                await updateCartItemQuantity(cartItemId, 1);
            }
        });

        itemElement.querySelector('.remove-from-cart-btn').addEventListener('click', async (e) => {
            const cartItemId = e.target.dataset.itemId;
            if (confirm('Are you sure you want to remove this item from your cart?')) {
                await removeFromCart(cartItemId);
            }
        });
    }

    async function updateCartItemQuantity(cartItemId, quantity) {
        try {
            await makeApiRequest(`/users/cart/${cartItemId}`, 'PUT', { quantity }, true);
            loadCart(); // Reload the cart to reflect changes
        } catch (error) {
            alert(`Failed to update quantity: ${error.message}`);
            loadCart(); // Reload to revert potential optimistic UI update
        }
    }

    async function removeFromCart(cartItemId) {
        try {
            await makeApiRequest(`/users/cart/${cartItemId}`, 'DELETE', null, true);
            loadCart(); // Reload cart
        } catch (error) {
            alert(`Failed to remove item: ${error.message}`);
        }
    }

    function updateCartSummary(cart) {
        if (!cartSummaryContainer) return;

        const subtotal = cart.reduce((sum, item) => sum + (item.product.price * item.quantity), 0);
        const taxRate = 0.05; // Example 5% tax
        const tax = subtotal * taxRate;
        const shipping = subtotal > 50 ? 0 : 10; // Example: free shipping over $50
        const total = subtotal + tax + shipping;

        cartSummaryContainer.innerHTML = `
            <h3>Order Summary</h3>
            <p><span>Subtotal:</span> <span>₹${subtotal.toFixed(2)}</span></p>
            <p><span>Estimated Tax:</span> <span>₹${tax.toFixed(2)}</span></p>
            <p><span>Shipping:</span> <span>₹${shipping.toFixed(2)}</span></p>
            <hr>
            <p class="total-price"><span>Total:</span> <span>₹${total.toFixed(2)}</span></p>
            <a href="checkout.html" class="btn btn-primary checkout-btn ${cart.length === 0 ? 'disabled' : ''}">Proceed to Checkout</a>
        `;
        if(cart.length === 0) {
            cartSummaryContainer.querySelector('.checkout-btn').style.pointerEvents = 'none';
            cartSummaryContainer.querySelector('.checkout-btn').style.opacity = '0.5';
        }
    }
});

// Public function to add to cart (can be called from product.js)
// Ensure this is defined globally or exported if using modules
async function globalAddToCart(productId, quantity = 1, size) {
    if (!getToken()) {
        alert('Please login to add items to your cart.');
        window.location.href = `login.html?redirect=${window.location.pathname}${window.location.search}`;
        return;
    }
    try {
        await makeApiRequest('/users/cart', 'POST', { productId, quantity, size }, true);
        alert('Product added to cart!');
        updateCartCount(); // From main.js
        if (window.location.pathname.includes('cart.html')) { // If on cart page, reload it
            // This direct call might conflict with the DOMContentLoaded of cart.js.
            // Better to use a custom event or have loadCart as a global function.
            // For simplicity now, let's assume cart.js's loadCart will handle it if exposed.
            if(typeof loadCart === 'function') loadCart();
        }
    } catch (error) {
        alert(`Failed to add to cart: ${error.message}`);
    }
}