// Thumbnail gallery functionality
const thumbnails = document.querySelectorAll('.thumbnail');
const mainImage = document.getElementById('mainImage');
const zoomIcon = document.getElementById('zoomIcon');
const modal = document.getElementById('imageModal');
const zoomedImage = document.getElementById('zoomedImage');
const closeBtn = document.querySelector('.close');

// Switch main image when thumbnail is clicked
thumbnails.forEach(thumb => {
    thumb.addEventListener('click', function () {
        // Update main image
        mainImage.src = this.getAttribute('data-fullsize');
        mainImage.alt = this.alt;

        // Update active thumbnail
        thumbnails.forEach(t => t.classList.remove('active'));
        this.classList.add('active');
    });
});

// Zoom functionality
zoomIcon.addEventListener('click', function () {
    zoomedImage.src = mainImage.src;
    zoomedImage.alt = mainImage.alt;
    modal.style.display = "block";
});

// Close modal
closeBtn.addEventListener('click', function () {
    modal.style.display = "none";
});

// Close when clicking outside image
window.addEventListener('click', function (event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
});

// Tab functionality
document.querySelectorAll('.tab-link').forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and contents
        document.querySelectorAll('.tab-link').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        // Add active class to clicked tab and corresponding content
        tab.classList.add('active');
        const tabId = tab.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
    });
});

// Size selection functionality
document.querySelectorAll('.size-option').forEach(size => {
    size.addEventListener('click', () => {
        document.querySelectorAll('.size-option').forEach(s => s.classList.remove('selected'));
        size.classList.add('selected');
    });
});