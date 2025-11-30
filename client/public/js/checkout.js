// client/public/js/checkout.js

document.addEventListener('DOMContentLoaded', () => {
    const checkoutForm = document.getElementById('checkout-form');
    const orderSummaryCheckout = document.getElementById('order-summary-checkout'); // To display items being ordered

    if (!getToken()) {
        // Redirect to login if not logged in and trying to access checkout
        window.location.href = `login.html?redirect=checkout.html`;
        return;
    }

    if (orderSummaryCheckout) {
        loadOrderSummaryForCheckout();
    }

    async function loadOrderSummaryForCheckout() {
        try {
            const cart = await makeApiRequest('/users/cart', 'GET', null, true);
            if (cart.length === 0) {
                // Redirect to cart page if cart is empty
                alert("Your cart is empty. Add items to proceed to checkout.");
                window.location.href = 'cart.html';
                return;
            }

            let summaryHTML = '<h4>Items in your order:</h4><ul>';
            let subtotal = 0;
            cart.forEach(item => {
                if (item.product) {
                    summaryHTML += `<li>${item.product.name} (x${item.quantity}) - Size: ${item.size} - ₹${(item.product.price * item.quantity).toFixed(2)}</li>`;
                    subtotal += item.product.price * item.quantity;
                }
            });
            summaryHTML += '</ul>';
            const tax = subtotal * 0.05; // Example tax
            const shipping = subtotal > 50 ? 0 : 10; // Example shipping
            const total = subtotal + tax + shipping;
            summaryHTML += `<p><strong>Subtotal: ₹${subtotal.toFixed(2)}</strong></p>`;
            summaryHTML += `<p><strong>Total (incl. tax & shipping): ₹${total.toFixed(2)}</strong></p>`;
            orderSummaryCheckout.innerHTML = summaryHTML;

        } catch (error) {
            orderSummaryCheckout.innerHTML = '<p class="error-message">Could not load order summary.</p>';
        }
    }


    if (checkoutForm) {
        checkoutForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Collect shipping address (ensure your form has these fields)
            const shippingAddress = {
                address: checkoutForm.address.value,
                city: checkoutForm.city.value,
                postalCode: checkoutForm.postalCode.value,
                country: checkoutForm.country.value,
            };

            // Dummy payment processing
            displayMessage('checkout-message', 'Processing payment...', false);
            showLoader('checkout-loader');

            try {
                // Simulate payment delay
                await new Promise(resolve => setTimeout(resolve, 1500));

                const orderData = {
                    shippingAddress,
                    // paymentMethod: 'DummyCard', // If you add payment method selection
                };

                const createdOrder = await makeApiRequest('/orders', 'POST', orderData, true);
                displayMessage('checkout-message', `Order placed successfully! Order ID: ${createdOrder._id}`, false);
                updateCartCount(); // Cart should be empty now
                // Redirect to order confirmation/history page
                setTimeout(() => {
                    window.location.href = `order-confirmation.html?orderId=${createdOrder._id}`; // Or order-history.html
                }, 2000);

            } catch (error) {
                displayMessage('checkout-message', `Order placement failed: ${error.message}`, true);
            } finally {
                hideLoader('checkout-loader');
            }
        });
    }
});