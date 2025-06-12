export const data=`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Navbar with Cards Grid</title>
  <style>
    /* Reset some default styles */
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background-color: #f4f4f4;
      color: #333;
    }

    /* Navbar styles */
    .navbar {
      background-color: #333;
      color: white;
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .navbar .logo {
      font-size: 1.5rem;
      font-weight: bold;
    }

    .navbar .menu {
      display: flex;
      gap: 1.5rem;
    }

    .navbar .menu a {
      color: white;
      text-decoration: none;
      font-size: 1rem;
      transition: color 0.3s;
    }

    .navbar .menu a:hover {
      color: #f0a500;
    }

    /* Card grid container */
    .card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 2rem;
      padding: 2rem;
    }

    /* Card styles */
    .card {
      background-color: white;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s;
    }

    .card:hover {
      transform: translateY(-5px);
    }

    .card img {
      width: 100%;
      height: 160px;
      object-fit: cover;
    }

    .card-body {
      padding: 1rem;
    }

    .card-title {
      font-size: 1.2rem;
      margin-bottom: 0.5rem;
    }

    .card-text {
      font-size: 0.95rem;
      color: #666;
    }
  </style>
</head>
<body>

  <!-- Navbar -->
  <nav class="navbar">
    <div class="logo">MySite</div>
    <div class="menu">
      <a href="#">Home</a>
      <a href="#">About</a>
      <a href="#">Services</a>
      <a href="#">Contact</a>
    </div>
  </nav>

  <!-- Card Grid -->
  <div class="card-grid">
    <!-- Card 1 -->
    <div class="card">
      <img src="https://via.placeholder.com/300x160" alt="Card Image">
      <div class="card-body">
        <h3 class="card-title">Card Title 1</h3>
        <p class="card-text">This is a short description of the first card. It looks nice and clean.</p>
      </div>
    </div>

    <!-- Card 2 -->
    <div class="card">
      <img src="https://via.placeholder.com/300x160" alt="Card Image">
      <div class="card-body">
        <h3 class="card-title">Card Title 2</h3>
        <p class="card-text">Here's some more text describing this second card in the grid layout.</p>
      </div>
    </div>

    <!-- Card 3 -->
    <div class="card">
      <img src="https://via.placeholder.com/300x160" alt="Card Image">
      <div class="card-body">
        <h3 class="card-title">Card Title 3</h3>
        <p class="card-text">Each card adapts nicely across different screen sizes.</p>
      </div>
    </div>

    <!-- Add more cards as needed -->
  </div>

</body>
</html>
`