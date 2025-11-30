// client/public/js/orders.js

document.addEventListener('DOMContentLoaded', () => {
    const orderHistoryContainer = document.getElementById('order-history-container');

    if (!getToken()) {
        // Redirect to login if not logged in
        window.location.href = `login.html?redirect=order-history.html`;
        return;
    }

    if (orderHistoryContainer) {
        loadOrderHistory();
    }

    async function loadOrderHistory() {
        showLoader('orders-loader');
        try {
            const orders = await makeApiRequest('/users/orders', 'GET', null, true);
            orderHistoryContainer.innerHTML = ''; // Clear previous
            if (orders.length === 0) {
                orderHistoryContainer.innerHTML = '<p>You have no past orders.</p>';
            } else {
                orders.forEach(order => displayOrder(order));
            }
        } catch (error) {
            orderHistoryContainer.innerHTML = `<p class="error-message">Failed to load order history: ${error.message}</p>`;
        } finally {
            hideLoader('orders-loader');
        }
    }

    function displayOrder(order) {
        const orderElement = document.createElement('div');
        orderElement.className = 'order-summary-item'; // Re-use cart item styling or create new
        let productsHTML = '<ul>';
        order.products.forEach(p => {
            productsHTML += `<li>${p.name} (x${p.quantity}) - Size: ${p.size} - Price: ₹${p.price.toFixed(2)}</li>`;
        });
        productsHTML += '</ul>';

        orderElement.innerHTML = `
            <div>
                <h4>Order ID: ${order._id}</h4>
                <p>Date: ${new Date(order.date).toLocaleDateString()}</p>
                <p>Total Price: ₹${order.totalPrice.toFixed(2)}</p>
                <p>Status: ${order.orderStatus} (Payment: ${order.paymentStatus})</p>
                <p>Shipping Address: ${order.shippingAddress.address}, ${order.shippingAddress.city}, ${order.shippingAddress.postalCode}</p>
                <h5>Items:</h5>
                ${productsHTML}
                <p><strong>Total: ₹${order.totalPrice.toFixed(2)}</strong></p>
            </div>
        `;
        // Add link to view order details if you have a specific order detail page
        // <a href="order-detail.html?id=${order._id}">View Details</a>
        orderHistoryContainer.appendChild(orderElement);
    }
});