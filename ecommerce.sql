-- phpMyAdmin SQL Dump
-- version 4.9.1
-- https://www.phpmyadmin.net/
--
-- Server: localhost
-- Generation Time: 18 Sep 2023, 17:24:10
-- Server version: 8.0.17
-- PHP Version: 7.3.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `ecommerce`
--

-- --------------------------------------------------------

--
-- Table structure for table `about`
--

CREATE TABLE `about` (
  `about_id` int(11) NOT NULL,
  `about_title` varchar(250) COLLATE utf8_unicode_ci NOT NULL,
  `about_content` text COLLATE utf8_unicode_ci NOT NULL,
  `about_video` varchar(50) COLLATE utf8_unicode_ci NOT NULL,
  `about_vision` varchar(500) COLLATE utf8_unicode_ci NOT NULL,
  `about_mission` varchar(500) COLLATE utf8_unicode_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `about`
--

INSERT INTO `about` (`about_id`, `about_title`, `about_content`, `about_video`, `about_vision`, `about_mission`) VALUES
(1, 'About Title', '<p><strong>Lorem ipsum dolor sit amet, consectetur adipiscing elit. </strong>Proin sed ex vel elit luctus euismod. Donec commodo a massa quis ultricies. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec condimentum placerat massa, eu ornare nibh sagittis sit amet. Nullam convallis cursus scelerisque. Ut iaculis sollicitudin dolor, vel lobortis leo tempus in. Donec rutrum justo id viverra convallis. Fusce gravida ullamcorper posuere.</p>', 'jNQXAC9IVRw', 'Nullam ut varius sapien. In quis urna ut felis hendrerit vehicula tincidunt eu mi. Donec varius varius sem at bibendum. Mauris placerat libero ut accumsan tempor. Praesent eu sodales urna. Morbi sed metus convallis, egestas nisl vel, tristique ante.', 'Nullam ut varius sapien. In quis urna ut felis hendrerit vehicula tincidunt eu mi. Donec varius varius sem at bibendum. Mauris placerat libero ut accumsan tempor. Praesent eu sodales urna. Morbi sed metus convallis, egestas nisl vel, tristique ante.');

-- --------------------------------------------------------

--
-- Table structure for table `bank`
--

CREATE TABLE `bank` (
  `bank_id` int(11) NOT NULL,
  `bank_name` varchar(100) COLLATE utf8_unicode_ci NOT NULL,
  `bank_iban` varchar(100) COLLATE utf8_unicode_ci NOT NULL,
  `bank_accountname` varchar(100) COLLATE utf8_unicode_ci NOT NULL,
  `bank_status` enum('0','1') COLLATE utf8_unicode_ci NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `bank`
--

INSERT INTO `bank` (`bank_id`, `bank_name`, `bank_iban`, `bank_accountname`, `bank_status`) VALUES
(1, 'ABC Bank', 'BN01001111000123', 'Name Surname', '1'),
(2, 'DEFG Bank', 'BN111122223334445', 'Namee Surname 2', '1'),
(4, 'QWERTY Bank', 'BN98763454353342', 'Name Surname 3', '1');

-- --------------------------------------------------------

--
-- Table structure for table `basket`
--

CREATE TABLE `basket` (
  `basket_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `product_id` int(11) NOT NULL,
  `product_unit` int(3) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `categories`
--

CREATE TABLE `categories` (
  `category_id` int(11) NOT NULL,
  `category_name` varchar(100) COLLATE utf8_unicode_ci NOT NULL,
  `category_top` int(2) NOT NULL,
  `category_seourl` varchar(250) COLLATE utf8_unicode_ci NOT NULL,
  `category_order` int(2) NOT NULL,
  `category_status` enum('0','1') COLLATE utf8_unicode_ci NOT NULL DEFAULT '1'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `categories`
--

INSERT INTO `categories` (`category_id`, `category_name`, `category_top`, `category_seourl`, `category_order`, `category_status`) VALUES
(1, 'Top', 0, 'top', 0, '1'),
(2, 'Outerwear', 0, 'outerwear', 1, '1'),
(3, 'Bottom', 0, 'bottom', 2, '1'),
(4, 'Accessories', 0, 'accessories', 3, '1'),
(5, 'Cardigan and Knitwear', 0, 'cardigan_and_knitwear', 4, '1'),
(6, 'Footwear', 0, 'footwear', 5, '1');

-- --------------------------------------------------------

--
-- Table structure for table `comments`
--

CREATE TABLE `comments` (
  `comment_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `product_id` int(11) NOT NULL,
  `comment_detail` text COLLATE utf8_unicode_ci NOT NULL,
  `comment_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `comment_status` enum('0','1') COLLATE utf8_unicode_ci NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `comments`
--

INSERT INTO `comments` (`comment_id`, `user_id`, `product_id`, `comment_detail`, `comment_time`, `comment_status`) VALUES
(1, 8, 9, 'test', '2023-07-02 20:04:09', '1'),
(2, 8, 9, 'test', '2023-07-02 20:12:53', '0'),
(3, 8, 9, 'test comment', '2023-07-02 20:30:26', '1'),
(4, 8, 9, 'Test Comment', '2023-07-06 19:25:30', '0'),
(6, 8, 7, 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam ut cursus turpis, id porta augue. Duis efficitur, leo sit amet luctus venenatis, risus sem accumsan justo, et fringilla magna nisi vitae lectus. Quisque euismod molestie viverra. Donec consectetur dui vitae justo pharetra malesuada. Praesent euismod lectus velit. Nullam dignissim finibus libero, eget aliquam justo volutpat eu. Nullam at feugiat augue. Fusce non laoreet metus, eget gravida magna.', '2023-07-07 20:11:56', '1');

-- --------------------------------------------------------

--
-- Table structure for table `menu`
--

CREATE TABLE `menu` (
  `menu_id` int(11) NOT NULL,
  `menu_top` varchar(50) COLLATE utf8_unicode_ci NOT NULL,
  `menu_name` varchar(100) COLLATE utf8_unicode_ci NOT NULL,
  `menu_detail` text COLLATE utf8_unicode_ci NOT NULL,
  `menu_url` varchar(250) COLLATE utf8_unicode_ci NOT NULL,
  `menu_order` int(2) NOT NULL,
  `menu_status` enum('0','1') COLLATE utf8_unicode_ci NOT NULL,
  `menu_seourl` varchar(250) COLLATE utf8_unicode_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `menu`
--

INSERT INTO `menu` (`menu_id`, `menu_top`, `menu_name`, `menu_detail`, `menu_url`, `menu_order`, `menu_status`, `menu_seourl`) VALUES
(1, '0', 'About', '', 'about', 0, '1', 'about'),
(3, '0', 'Contact', '<p>Contact Page Content</p>', 'contact', 3, '1', 'contact'),
(4, '0', 'Categories', '<p><strong>Lorem Ipsum</strong> is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry\'s standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.</p>', 'categories', 1, '1', 'categories'),
(7, '', 'Landing', '<p>Landing Page</p>', '', 4, '1', 'landing');

-- --------------------------------------------------------

--
-- Table structure for table `orders`
--

CREATE TABLE `orders` (
  `order_id` int(11) NOT NULL,
  `order_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `order_no` int(11) DEFAULT NULL,
  `user_id` int(11) NOT NULL,
  `order_total` float(9,2) NOT NULL,
  `order_type` varchar(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,
  `order_bank` varchar(50) COLLATE utf8_unicode_ci NOT NULL,
  `order_pay` enum('0','1') CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `orders`
--

INSERT INTO `orders` (`order_id`, `order_time`, `order_no`, `user_id`, `order_total`, `order_type`, `order_bank`, `order_pay`) VALUES
(1003, '2023-07-25 19:59:52', NULL, 8, 9000.00, 'Bank Payment', 'DEFG Bank', '0'),
(1004, '2023-07-25 20:06:53', NULL, 8, 750.00, 'Bank Payment', 'ABC Bank', '0'),
(1005, '2023-08-12 11:46:15', NULL, 8, 583.00, 'Bank Payment', 'DEFG Bank', '0');

-- --------------------------------------------------------

--
-- Table structure for table `orders_detail`
--

CREATE TABLE `orders_detail` (
  `orderdetail_id` int(11) NOT NULL,
  `order_id` int(11) NOT NULL,
  `product_id` int(11) NOT NULL,
  `product_price` float(9,2) NOT NULL,
  `product_unit` int(4) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `orders_detail`
--

INSERT INTO `orders_detail` (`orderdetail_id`, `order_id`, `product_id`, `product_price`, `product_unit`) VALUES
(7, 1003, 1, 375.00, 1),
(8, 1003, 5, 375.00, 15),
(9, 1003, 5, 375.00, 1),
(10, 1003, 5, 375.00, 5),
(11, 1003, 5, 375.00, 1),
(12, 1003, 5, 375.00, 1),
(13, 1004, 1, 375.00, 2),
(14, 1005, 7, 583.00, 1);

-- --------------------------------------------------------

--
-- Table structure for table `product`
--

CREATE TABLE `product` (
  `product_id` int(11) NOT NULL,
  `category_id` int(11) NOT NULL,
  `product_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `product_name` varchar(250) COLLATE utf8_unicode_ci NOT NULL,
  `product_seourl` varchar(250) COLLATE utf8_unicode_ci NOT NULL,
  `product_detail` text COLLATE utf8_unicode_ci NOT NULL,
  `product_price` float(9,2) NOT NULL,
  `product_video` varchar(50) COLLATE utf8_unicode_ci NOT NULL,
  `product_keyword` varchar(250) COLLATE utf8_unicode_ci NOT NULL,
  `product_stock` int(11) NOT NULL,
  `product_status` enum('0','1') COLLATE utf8_unicode_ci NOT NULL,
  `product_order` int(2) NOT NULL,
  `product_homeshowcase` enum('0','1') COLLATE utf8_unicode_ci NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `product`
--

INSERT INTO `product` (`product_id`, `category_id`, `product_time`, `product_name`, `product_seourl`, `product_detail`, `product_price`, `product_video`, `product_keyword`, `product_stock`, `product_status`, `product_order`, `product_homeshowcase`) VALUES
(1, 2, '2023-05-12 19:45:10', 'Off Shoulder', 'off-shoulder', '<p>An off-shoulder top is a trendy garment that leaves the shoulders bare, with a neckline that sits below the collarbones. It can be fitted or flowy, with various sleeve lengths, and often features elastic or ruffled details. Perfect for adding a touch of elegance to casual or dressy outfits.</p>', 75.00, 'jNQXAC9IVRw', 'product, price, stock, name ', 125, '1', 1, '1'),
(2, 2, '2023-05-15 20:25:44', 'V Neck Shirt', 'v-neck-shirt', '<p>A V-neck top features a neckline that dips into a V-shape at the front, elongating the neck and creating a flattering, streamlined look. This versatile style can range from casual to formal and comes in various fits and sleeve lengths, making it a great choice for any occasion.</p>', 50.00, 'test', 'product, price, stock, name ', 8, '1', 2, '1'),
(3, 6, '2023-05-15 20:27:41', 'Cropped Jacket', 'cropped-jacket', '<p>A cropped jacket is a shorter-length jacket that typically ends above the waist, giving a modern and edgy vibe. It’s designed to highlight the waistline and can come in various styles, from denim and leather to tailored cuts. This versatile piece adds a chic layer to both casual and dressed-up outfits, perfect for a trendy, fashion-forward look.</p>', 75.00, '', 'product, price, stock, name ', 0, '1', 3, '0'),
(4, 9, '2023-05-12 19:45:10', 'Wrap Dress', 'wrap-dress', '<p>A wrap dress is a flattering and versatile garment designed with a front closure that wraps around the body, usually tying at the side. This creates a V-shaped neckline and a fitted waist, making it adjustable and comfortable. Known for its elegant silhouette, the wrap dress works well for both casual and formal occasions, offering a timeless and feminine look.</p>', 100.00, 'test', 'product, price, stock, name ', 125, '1', 4, '1'),
(5, 2, '2023-05-15 20:25:44', 'Polo Shirt', 'polo-shirt', '<p>A polo shirt is a classic, short-sleeved top featuring a soft collar, a buttoned placket, and a slightly fitted shape. Traditionally made from breathable cotton or pique fabric, it combines comfort with a sporty, polished look. Perfect for both casual and semi-formal settings, the polo shirt is a versatile wardrobe staple.</p>', 50.00, 'test', 'product, price, stock, name ', 8, '1', 5, '0'),
(6, 5, '2023-05-15 20:27:41', 'Chinos', 'chinos', '<p>Chinos are lightweight, cotton-blend trousers known for their casual yet polished appearance. They feature a slim, straight-leg cut and are typically designed with minimal detailing for a clean, streamlined look. Versatile and comfortable, chinos can be dressed up or down, making them perfect for both casual and smart-casual occasions.</p>', 100.00, 'test', 'product, price, stock, name ', 5, '1', 6, '1'),
(7, 5, '2023-05-12 19:45:10', 'Slim Fit Trouser', 'slim-fit-trouser', '<p>Slim-fit trousers are tailored pants designed to fit closely to the body, creating a sleek and modern silhouette. They feature a tapered leg that narrows from the thigh to the ankle, offering a streamlined look without being overly tight. Made from various fabrics, these trousers provide both comfort and style, making them suitable for both casual and formal settings.</p>', 100.00, 'test', 'product, price, stock, name ', 125, '1', 7, '0'),
(8, 9, '2023-05-15 20:25:44', 'A-Line Dress', 'a-line-dress', '<p>A-line dresses are characterized by their fitted bodice and flared skirt that gradually widens from the waist down, resembling the shape of a capital "A." This flattering silhouette enhances the waist and allows for ease of movement, making them suitable for a variety of body types. A-line dresses come in various lengths and styles, from casual to formal, making them a versatile choice for any occasion.</p>', 150.00, 'test', 'product, price, stock, name ', 8, '1', 8, '1'),
(9, 6, '2023-05-15 20:27:41', 'Bomber Jacket', 'bomber-jackets', '<p>A bomber jacket is a casual, lightweight outerwear piece originally designed for pilots. Characterized by its fitted waist and cuffs, it features a zippered front, a round neckline, and often comes with practical pockets. Made from various materials such as nylon, polyester, or leather, the bomber jacket offers a relaxed yet stylish silhouette.</p>', 150.00, 'test', 'product, price, stock, name ', 5, '1', 9, '1'),
(10, 6, '2023-05-12 19:45:10', 'Denim Jacket', 'denim-jacket', '<p>A denim jacket is a timeless outerwear staple made from sturdy cotton twill fabric, known for its durability and versatility. Typically featuring a button-up front, chest pockets, and a classic collar, it offers a relaxed yet stylish look. The jacket often has a slightly fitted or oversized silhouette, making it easy to layer over various outfits. Available in different washes and colors, the denim jacket can be dressed up or down, making it perfect for casual outings or as a stylish layer in cooler weather.</p>', 200.00, 'test', 'product, price, stock, name ', 125, '1', 10, '0'),
(11, 5, '2023-05-15 20:25:44', 'High-Waisted Pants', 'high-waisted-pants', '<p>High-waisted pants are designed to sit above the natural waistline, creating an elongated silhouette and accentuating the waist. They come in various styles, including wide-leg, straight-leg, or fitted cuts, and are often made from materials like denim, cotton, or trousers. These pants are versatile, easily paired with cropped tops or tucked-in blouses for a chic look, making them a popular choice for both casual and formal occasions. High-waisted pants offer comfort and style while enhancing the bodys proportions.</p>', 75.00, 'test', 'product, price, stock, name ', 8, '1', 11, '1'),
(12, 9, '2023-05-15 20:27:41', 'Maxi Dress', 'maxi-dress', '<p>Maxi dresses are long, flowing garments that typically extend to the ankles or floor, offering a relaxed and comfortable fit. Characterized by their sleeveless or short-sleeved designs, they often feature a variety of necklines, patterns, and materials, making them suitable for both casual and formal occasions. Maxi dresses are perfect for warm weather, providing an effortlessly chic look that flatters a range of body types and can be accessorized for different styles, from bohemian to elegant.</p>', 250.00, 'test', 'product, price, stock, name ', 5, '1', 12, '1');

-- --------------------------------------------------------

--
-- Table structure for 'product_variant'
--

CREATE TABLE `product_variant` (
  `variant_id` int(11) AUTO_INCREMENT PRIMARY KEY,
  `product_id` int(11) NOT NULL,
  `size` varchar(50) NOT NULL,
  `color` varchar(50) NOT NULL,
  `stock` int(11) NOT NULL,
  `product_price` float(9,2) NOT NULL,
  FOREIGN KEY (product_id) REFERENCES product(product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `product_variant`
--

INSERT INTO `product_variant` (`variant_id`, `product_id`, `size`, `color`, `stock`, `product_price`) VALUES
(1, 1, 'S', 'Black', 15, 75),
(2, 1, 'M', 'Black', 20, 75),
(3, 1, 'L', 'Black', 20, 75),
(4, 1, 'XL', 'Black', 20, 75),
(5, 1, 'S', 'White', 20, 75),
(6, 1, 'M', 'White', 20, 75),
(7, 1, 'L', 'White', 20, 75),
(8, 1, 'XL', 'White', 20, 75);

--
-- Table structure fot `product_photo`
--

CREATE TABLE `product_photo` (
  `productphoto_id` int(11) NOT NULL AUTO_INCREMENT,
  `product_id` int(11) NOT NULL,
  `productphoto_path` varchar(255) CHARACTER SET utf8 COLLATE utf8_unicode_ci NOT NULL,
  `productphoto_order` int(3) NOT NULL,
  `color` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `product_photo`
--

INSERT INTO `product_photo` (`productphoto_id`, `product_id`, `productphoto_path`, `productphoto_order`, `color`) VALUES
(1, 1, 'images/product/off-shoulder.jpg', 1, 'Black'),
(2, 2, 'images/product/v-neck-shirt.jpg', 2, 'Blue'),
(3, 3, 'images/product/cropped-jacket.jpg', 3, 'Black'),
(4, 4, 'images/product/wrap-dress.jpg', 4, 'Red'),
(5, 5, 'images/product/polo-shirt.jpg', 5, 'White'),
(6, 6, 'images/product/chinos.jpg', 6, 'Khaki'),
(7, 7, 'images/product/slim-fit-pants.jpg', 0, 'Grey'),
(8, 8, 'images/product/a-line-dress.jpg', 0, 'Green'),
(9, 9, 'images/product/bomber-jacket.jpg', 0, 'Black'),
(10, 10, 'images/product/denim-jacket.jpg', 0, 'Blue'),
(11, 11, 'images/product/high-waisted-pants.jpg', 0, 'Brown'),
(12, 12, 'images/product/maxi-dress.jpg', 0, 'Yellow'),
(13, 13, 'images/product/28490275702375724936208952749925167236186.jpg', 0, 'Pink'),
(14, 5, 'images/product/polo-shirt.jpg', 0, 'Blue'),
(15, 4, 'images/product/wrap-dress.jpg', 0, 'Black'),
(16, 3, 'images/product/cropped-jacket.jpg', 0, 'Grey'),
(17, 1, 'images/product/off-shoulder-white.jpg', 0, 'White');

-- --------------------------------------------------------

--
-- Tablo structure for table `setting`
--

CREATE TABLE `setting` (
  `setting_id` int(11) NOT NULL,
  `setting_logo` varchar(150) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_url` varchar(50) COLLATE utf8_unicode_ci NOT NULL,
  `setting_title` varchar(250) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_description` varchar(250) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_keywords` varchar(250) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_author` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_tel` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_gsm` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_fax` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_mail` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_district` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_country` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_adress` varchar(250) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_time` varchar(250) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_maps` varchar(250) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_analystic` varchar(250) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_desk` varchar(250) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_facebook` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_twitter` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_google` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_youtube` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_smtphost` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_smtppassword` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_smtpport` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `setting_live` enum('0','1') COLLATE utf8_unicode_ci NOT NULL DEFAULT '1'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `setting`
--

INSERT INTO `setting` (`setting_id`, `setting_logo`, `setting_url`, `setting_title`, `setting_description`, `setting_keywords`, `setting_author`, `setting_tel`, `setting_gsm`, `setting_fax`, `setting_mail`, `setting_district`, `setting_country`, `setting_adress`, `setting_time`, `setting_maps`, `setting_analystic`, `setting_desk`, `setting_facebook`, `setting_twitter`, `setting_google`, `setting_youtube`, `setting_smtphost`, `setting_smtppassword`, `setting_smtpport`, `setting_live`) VALUES
(1, 'images/30283logo.png', 'http://siteadi.com', 'E-Commerce Page', 'E-commerce shopping page', 'E-Commerce, Shopping', 'MBCK', '02120000000', '05000000000', '02160000000', 'admin@ecommerce.com', 'Kadıköy', 'İstanbul', 'Bağdat Cad., No:2B, Kadıköy, İstanbul, Türkiye', '09:00 - 18:00', 'Maps test code', 'Analystic test code', 'Helpdesk test code', 'facebook_adresi', 'twitter_adresi', 'google_adresi', 'youtube_adresi', 'host', '123456789', 'port', '1');

-- --------------------------------------------------------

--
-- Tablo structure for table `slider`
--

CREATE TABLE `slider` (
  `slider_id` int(11) NOT NULL,
  `slider_name` varchar(100) COLLATE utf8_unicode_ci NOT NULL,
  `slider_imgurl` varchar(250) COLLATE utf8_unicode_ci NOT NULL,
  `slider_order` int(2) NOT NULL,
  `slider_link` varchar(250) COLLATE utf8_unicode_ci NOT NULL,
  `slider_status` enum('0','1') COLLATE utf8_unicode_ci NOT NULL DEFAULT '1'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `slider`
--

INSERT INTO `slider` (`slider_id`, `slider_name`, `slider_imgurl`, `slider_order`, `slider_link`, `slider_status`) VALUES
(5, 'Slider 1', 'images/slider/28466236822818125814slide-1.jpg', 1, 'test', '1'),
(6, 'Slider 2', 'images/slider/24799289202516622533slide-2.jpg', 2, '', '1'),
(7, 'Slider 3', 'images/slider/30063252702659021852slide-3.jpg', 3, '', '1'),
(10, 'Slider 4', 'images/slider/29862204183105924319slide-4.jpg', 4, 'test', '1');

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `user_id` int(11) NOT NULL,
  `user_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `user_image` varchar(250) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_tc` varchar(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_mail` varchar(100) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_gsm` varchar(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_password` varchar(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_adress` varchar(250) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_country` varchar(100) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_district` varchar(100) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_title` varchar(100) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_authority` varchar(50) CHARACTER SET utf8 COLLATE utf8_unicode_ci DEFAULT NULL,
  `user_status` int(1) NOT NULL DEFAULT '1'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- Table dump data `user`
--

INSERT INTO `user` (`user_id`, `user_time`, `user_image`, `user_tc`, `user_name`, `user_mail`, `user_gsm`, `user_password`, `user_adress`, `user_country`, `user_district`, `user_title`, `user_authority`, `user_status`) VALUES
(1, '2023-04-19 19:17:15', NULL, NULL, 'Name', 'name.surname@ecommerce.com', '0500 000 00 00', '25d55ad283aa400af464c76d713c07ad', 'Türkiye', 'İstanbul', 'Üsküdar', NULL, '9', 1),
(2, '2023-04-19 19:17:15', NULL, NULL, 'Nameee', '1@ecommerce.com', '0500 000 00 00', '25d55ad283aa400af464c76d713c07ad', 'Türkiye', 'İstanbul', 'Üsküdar', NULL, '9', 1),
(3, '2023-04-19 19:17:15', NULL, NULL, 'Name123', '2@ecommerce.com', '0500 000 00 00', '25d55ad283aa400af464c76d713c07ad', 'Türkiye', 'İstanbul', 'Üsküdar', NULL, '9', 1),
(8, '2023-05-02 19:47:37', NULL, NULL, 'test test', 'test@test.com', '0500 000 00 00', '25d55ad283aa400af464c76d713c07ad', 'Türkiye', 'İstanbul', 'Fatih', NULL, '1', 1),
(9, '2023-05-05 15:10:52', NULL, NULL, 'test1', 'test1@test.com', NULL, 'e99a18c428cb38d5f260853678922e03', NULL, NULL, NULL, NULL, '1', 1);

--
-- Indexes for dumped tables
--

--
-- Indexes for the table `about`
--
ALTER TABLE `about`
  ADD PRIMARY KEY (`about_id`);

--
-- Indexes for the table `bank`
--
ALTER TABLE `bank`
  ADD PRIMARY KEY (`bank_id`);

--
-- Indexes for the table `basket`
--
ALTER TABLE `basket`
  ADD PRIMARY KEY (`basket_id`);

--
-- Indexes for the table `categories`
--
ALTER TABLE `categories`
  ADD PRIMARY KEY (`category_id`);

--
-- Indexes for the table `comments`
--
ALTER TABLE `comments`
  ADD PRIMARY KEY (`comment_id`);

--
-- Indexes for the table `menu`
--
ALTER TABLE `menu`
  ADD PRIMARY KEY (`menu_id`);

--
-- Indexes for the table `orders`
--
ALTER TABLE `orders`
  ADD PRIMARY KEY (`order_id`);

--
-- Indexes for the table `orders_detail`
--
ALTER TABLE `orders_detail`
  ADD PRIMARY KEY (`orderdetail_id`);

--
-- Indexes for the table `product`
--
ALTER TABLE `product`
  ADD PRIMARY KEY (`product_id`);

--
-- Indexes for the table `product_photo`
--
ALTER TABLE `product_photo`
  ADD PRIMARY KEY (`productphoto_id`);

--
-- Indexes for the table `setting`
--
ALTER TABLE `setting`
  ADD PRIMARY KEY (`setting_id`);

--
-- Indexes for the table `slider`
--
ALTER TABLE `slider`
  ADD PRIMARY KEY (`slider_id`);

--
-- Indexes for the table `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`user_id`);

--
-- AUTO_INCREMENT value for dumped tables
--

--
-- AUTO_INCREMENT value for table `bank`
--
ALTER TABLE `bank`
  MODIFY `bank_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT value for table `basket`
--
ALTER TABLE `basket`
  MODIFY `basket_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT value for table `categories`
--
ALTER TABLE `categories`
  MODIFY `category_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;

--
-- AUTO_INCREMENT value for table `comments`
--
ALTER TABLE `comments`
  MODIFY `comment_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- AUTO_INCREMENT value for table `menu`
--
ALTER TABLE `menu`
  MODIFY `menu_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT value for table `orders`
--
ALTER TABLE `orders`
  MODIFY `order_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1006;

--
-- AUTO_INCREMENT value for table `orders_detail`
--
ALTER TABLE `orders_detail`
  MODIFY `orderdetail_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;

--
-- AUTO_INCREMENT value for table `product`
--
ALTER TABLE `product`
  MODIFY `product_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;

--
-- AUTO_INCREMENT value for table `product_photo`
--
ALTER TABLE `product_photo`
  MODIFY `productphoto_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=21;

--
-- AUTO_INCREMENT value for table `setting`
--
ALTER TABLE `setting`
  MODIFY `setting_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT value for table `slider`
--
ALTER TABLE `slider`
  MODIFY `slider_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT value for table `user`
--
ALTER TABLE `user`
  MODIFY `user_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
