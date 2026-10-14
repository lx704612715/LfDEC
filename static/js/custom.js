/* Click-to-play YouTube thumbnails for the Multimedia Extensions grid.
   Nothing is requested from YouTube until the reader presses play. */
(function () {
  'use strict';
  function play(btn) {
    var id = btn.getAttribute('data-id');
    if (!id) return;
    var frame = document.createElement('iframe');
    frame.src = 'https://www.youtube-nocookie.com/embed/' + encodeURIComponent(id) + '?autoplay=1&rel=0';
    frame.title = btn.getAttribute('aria-label') || 'Video';
    frame.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
    frame.setAttribute('allowfullscreen', '');
    frame.setAttribute('referrerpolicy', 'strict-origin-when-cross-origin');
    frame.className = 'yt playing';
    btn.replaceWith(frame);
    frame.focus();
  }
  document.addEventListener('click', function (ev) {
    var btn = ev.target.closest('.yt');
    if (btn && !btn.classList.contains('playing')) {
      ev.preventDefault();
      play(btn);
    }
  });
})();
