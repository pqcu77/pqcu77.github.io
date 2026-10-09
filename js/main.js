(function () {
  'use strict';

  /* ---------- 明暗切换 ---------- */
  var root = document.documentElement;
  var toggle = document.getElementById('themeToggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('ink-theme', next); } catch (e) {}
    });
  }

  /* ---------- 滚动进度条 & 回到顶部 ---------- */
  var bar = document.getElementById('progressBar');
  var top = document.getElementById('backToTop');
  var ticking = false;
  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    var y = h.scrollTop || document.body.scrollTop;
    if (bar && max > 0) bar.style.width = (y / max) * 100 + '%';
    if (top) top.classList.toggle('show', y > 400);
    ticking = false;
  }
  window.addEventListener('scroll', function () {
    if (!ticking) { ticking = true; window.requestAnimationFrame(onScroll); }
  }, { passive: true });
  onScroll();

  if (top) {
    top.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  /* ---------- 代码块复制 ---------- */
  var copyEnabled = document.body.classList.contains('page-post') ||
                    document.body.classList.contains('page-single');
  if (copyEnabled) {
    document.querySelectorAll('.highlight').forEach(function (fig) {
      var btn = document.createElement('button');
      btn.className = 'copy-code-btn';
      btn.type = 'button';
      btn.textContent = '复制';
      btn.addEventListener('click', function () {
        var codeEl = fig.querySelector('.code') || fig.querySelector('code');
        if (!codeEl) return;
        var text = codeEl.innerText || codeEl.textContent || '';
        var done = function () {
          btn.textContent = '已复制';
          btn.classList.add('done');
          setTimeout(function () {
            btn.textContent = '复制';
            btn.classList.remove('done');
          }, 1600);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done).catch(function () {});
        } else {
          var ta = document.createElement('textarea');
          ta.value = text;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand('copy'); done(); } catch (e) {}
          document.body.removeChild(ta);
        }
      });
      fig.appendChild(btn);
    });
  }

  /* ---------- 目录生成 + 高亮 ---------- */
  var tocList = document.getElementById('tocList');
  var body = document.getElementById('postBody');
  if (tocList && body) {
    var heads = body.querySelectorAll('h2, h3');
    if (heads.length >= 2) {
      var ul = document.createElement('ul');
      var slugCount = {};
      heads.forEach(function (h) {
        if (!h.id) {
          var base = (h.textContent || '').trim()
            .replace(/\s+/g, '-')
            .replace(/[^\w\u4e00-\u9fa5-]/g, '');
          if (!base) base = 'section';
          slugCount[base] = (slugCount[base] || 0) + 1;
          h.id = slugCount[base] > 1 ? base + '-' + slugCount[base] : base;
        }
        var li = document.createElement('li');
        li.className = 'toc-' + h.tagName.toLowerCase();
        var a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent;
        li.appendChild(a);
        ul.appendChild(li);
      });
      tocList.appendChild(ul);

      var links = Array.prototype.slice.call(tocList.querySelectorAll('a'));
      var targets = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });
      function syncToc() {
        var pos = window.scrollY + 120;
        var idx = -1;
        for (var i = 0; i < targets.length; i++) {
          if (targets[i] && targets[i].offsetTop <= pos) idx = i; else break;
        }
        links.forEach(function (a, i) { a.classList.toggle('active', i === idx); });
      }
      window.addEventListener('scroll', syncToc, { passive: true });
      syncToc();
    } else {
      var aside = document.getElementById('toc');
      if (aside) aside.style.display = 'none';
    }
  }

  /* ---------- 搜索 ---------- */
  var modal = document.getElementById('searchModal');
  var openBtn = document.getElementById('searchBtn');
  var input = document.getElementById('searchInput');
  var results = document.getElementById('searchResults');
  var index = null;

  if (modal && openBtn && input && results) {
    var searchPath = modal.getAttribute('data-search-path') || '/search.json';

    function open() {
      modal.hidden = false;
      input.value = '';
      results.innerHTML = '<div class="search-tip">输入关键词开始搜索…</div>';
      setTimeout(function () { input.focus(); }, 20);
      if (!index) {
        fetch(searchPath)
          .then(function (r) { return r.json(); })
          .then(function (data) { index = data; })
          .catch(function () { index = []; });
      }
    }
    function close() { modal.hidden = true; }

    openBtn.addEventListener('click', open);
    modal.addEventListener('click', function (e) {
      if (e.target.hasAttribute('data-close')) close();
    });
    document.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); modal.hidden ? open() : close(); }
      if (e.key === 'Escape' && !modal.hidden) close();
      if (e.key === '/' && document.activeElement !== input &&
          !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
        e.preventDefault(); open();
      }
    });

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
      });
    }
    function highlight(text, kw) {
      var safe = escapeHtml(text);
      if (!kw) return safe;
      var i = safe.toLowerCase().indexOf(kw.toLowerCase());
      if (i < 0) return safe;
      return safe.slice(0, i) + '<mark>' + safe.slice(i, i + kw.length) + '</mark>' + safe.slice(i + kw.length);
    }
    function render(kw) {
      if (!index) return;
      var q = kw.trim().toLowerCase();
      if (!q) {
        results.innerHTML = '<div class="search-tip">输入关键词开始搜索…</div>';
        return;
      }
      var hits = [];
      for (var i = 0; i < index.length && hits.length < 30; i++) {
        var p = index[i];
        var score = 0;
        var title = (p.title || '').toLowerCase();
        var tags = (p.tags || []).join(' ').toLowerCase();
        var text = (p.text || '').toLowerCase();
        if (title.indexOf(q) > -1) score += 10;
        if (tags.indexOf(q) > -1) score += 5;
        var ti = text.indexOf(q);
        if (ti > -1) score += 2;
        if (score) {
          var snippet = '';
          if (ti > -1) {
            var start = Math.max(0, ti - 30);
            snippet = (start > 0 ? '…' : '') + p.text.slice(start, start + 110);
          }
          hits.push({ p: p, snippet: snippet });
        }
      }
      if (!hits.length) {
        results.innerHTML = '<div class="search-empty">没有找到匹配的文章</div>';
        return;
      }
      results.innerHTML = hits.map(function (h) {
        return '<a class="search-hit" href="' + h.p.url + '">' +
          '<div class="search-hit-title">' + highlight(h.p.title, q) + '</div>' +
          (h.snippet ? '<div class="search-hit-desc">' + highlight(h.snippet, q) + '</div>' : '') +
          '</a>';
      }).join('');
    }

    var timer = null;
    input.addEventListener('input', function () {
      var v = input.value;
      clearTimeout(timer);
      timer = setTimeout(function () { render(v); }, 120);
    });
  }
})();
