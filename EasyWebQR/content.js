function initQRCode() {
  console.log("initQRCode function called");

  // 创建容器
  const container = document.createElement('div');
  container.id = 'qr-code-container';
  document.body.appendChild(container);

  // 获取当前页面URL和网站名称
  const currentUrl = window.location.href;
  const siteName = new URL(currentUrl).hostname;
  const pageTitle = document.title.length > 15 ? document.title.substring(0, 15) + '...' : document.title;

  // 创建图标元素
  const iconElement = document.createElement('div');
  iconElement.id = 'site-icon';
  container.appendChild(iconElement);

  // 加载网站图标
  loadSiteIcon(iconElement, siteName, currentUrl);

  // 创建二维码元素（初始隐藏）
  const qrWrapper = document.createElement('div');
  qrWrapper.id = 'qr-wrapper';
  qrWrapper.style.display = 'none';
  container.appendChild(qrWrapper);

  const qrElement = document.createElement('div');
  qrElement.id = 'qr-code';
  qrWrapper.appendChild(qrElement);

  // 创建信息容器
  const infoContainer = document.createElement('div');
  infoContainer.id = 'qr-info-container';
  qrWrapper.appendChild(infoContainer);

  // 创建网站名称元素
  const siteNameElement = document.createElement('div');
  siteNameElement.className = 'site-name';
  siteNameElement.textContent = siteName;
  infoContainer.appendChild(siteNameElement);

  // 创建网页标题元素
  const pageTitleElement = document.createElement('div');
  pageTitleElement.className = 'page-title';
  pageTitleElement.textContent = pageTitle;
  infoContainer.appendChild(pageTitleElement);

  // 点击图标时切换二维码显示
  iconElement.addEventListener('click', () => {
    if (qrWrapper.style.display === 'none') {
      qrWrapper.style.display = 'block';
      if (!qrElement.hasChildNodes()) {
        generateQRCode(qrElement, currentUrl, siteName);
      }
    } else {
      qrWrapper.style.display = 'none';
    }
  });

  // 添加右键下载功能
  qrWrapper.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const qrCanvas = qrElement.querySelector('canvas');
    if (qrCanvas) {
      downloadQRCode(qrCanvas, siteName, pageTitle);
    }
  });

  console.log("QR code container setup completed");
}

function loadSiteIcon(iconElement, siteName, currentUrl) {
  const iconPaths = [
    '/favicon.ico',
    '/favicon.png',
    '/apple-touch-icon.png',
    '/apple-touch-icon-precomposed.png'
  ];

  function tryNextIcon(index) {
    if (index >= iconPaths.length) {
      useFallbackIcon();
      return;
    }

    const iconUrl = new URL(iconPaths[index], currentUrl).href;
    fetch(iconUrl)
      .then(response => response.ok ? iconUrl : Promise.reject())
      .then(setIconBackground)
      .catch(() => tryNextIcon(index + 1));
  }

  function useFallbackIcon() {
    const fallbackUrl = `https://www.google.com/s2/favicons?domain=${siteName}&sz=64`;
    setIconBackground(fallbackUrl);
  }

  function setIconBackground(url) {
    iconElement.style.backgroundImage = `url('${url}')`;
    iconElement.style.backgroundSize = 'cover';
  }

  tryNextIcon(0);
}

function generateQRCode(element, url, siteName) {
  if (typeof QRCode === 'undefined') {
    console.error("QRCode library is not loaded");
    return;
  }

  try {
    new QRCode(element, {
      text: url,
      width: 224,  // 减小二维码尺寸，为logo留出空间
      height: 224,
      colorDark: "#000000",
      colorLight: "#ffffff",
      correctLevel: QRCode.CorrectLevel.H
    });

    // 在二维码生成后添加logo
    setTimeout(() => {
      const qrCanvas = element.querySelector('canvas');
      if (qrCanvas) {
        addLogo(qrCanvas, siteName, url);
      }
    }, 100);

    console.log("QR code generated successfully");
  } catch (error) {
    console.error("Error generating QR code:", error);
  }
}

function addLogo(canvas, siteName, url) {
  const ctx = canvas.getContext('2d');
  const logoSize = 48;  // 调整logo大小
  const logoX = (224 - logoSize) / 2;
  const logoY = (224 - logoSize) / 2;

  const favicon = new Image();
  favicon.crossOrigin = "Anonymous";
  favicon.src = new URL('/favicon.ico', url).href;  // 尝试获取网站的默认图标

  favicon.onload = function() {
    ctx.fillStyle = 'white';
    ctx.fillRect(logoX - 2, logoY - 2, logoSize + 4, logoSize + 4);
    ctx.drawImage(favicon, logoX, logoY, logoSize, logoSize);
  };

  favicon.onerror = function() {
    // 如果默认图标加载失败，尝试使用Google的服务
    favicon.src = `https://www.google.com/s2/favicons?domain=${siteName}&sz=64`;
  };
}

function downloadQRCode(canvas, siteName, pageTitle) {
  const downloadCanvas = document.createElement('canvas');
  downloadCanvas.width = 300;  // 增加画布宽度
  downloadCanvas.height = 340;  // 增加画布高度

  const ctx = downloadCanvas.getContext('2d');

  // 填充白色背景
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, 300, 340);

  // 绘制二维码（居中）
  ctx.drawImage(canvas, 38, 16);  // 38 = (300 - 224) / 2, 16是顶部留白

  // 绘制网站名称和页面标题
  ctx.font = '14px Arial';
  ctx.fillStyle = 'black';
  ctx.textAlign = 'center';
  ctx.fillText(siteName, 150, 274);
  ctx.font = 'bold 16px Arial';
  ctx.fillText(pageTitle, 150, 300);

  // 创建下载链接
  const link = document.createElement('a');
  
  // 使用当前时间戳创建文件名
  const timestamp = new Date().getTime();
  const fileName = `qrcode_${timestamp}.png`;
  
  link.download = fileName;
  link.href = downloadCanvas.toDataURL('image/png');
  link.click();
}

// 使用MutationObserver确保DOM已经准备好
const observer = new MutationObserver((mutations, obs) => {
  const body = document.querySelector('body');
  if (body) {
    console.log("Body element found, initializing QR code");
    setTimeout(initQRCode, 1000); // 延迟1秒执行
    obs.disconnect();
  }
});

observer.observe(document, {
  childList: true,
  subtree: true
});

console.log("Content script loaded");