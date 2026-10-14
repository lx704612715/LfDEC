(function () {
  document.querySelectorAll('.example-slider').forEach(function (slider) {
    var current = 0;
    var examples = JSON.parse(slider.querySelector('.slider-data').textContent);

    slider.addEventListener('click', function (event) {
      var arrow = event.target.closest('[data-example-step]');
      if (!arrow) return;
      current = (current + Number(arrow.getAttribute('data-example-step')) + examples.length) % examples.length;
      var example = examples[current];
      var video = document.createElement('video');
      video.controls = true;
      video.playsInline = true;
      video.preload = 'none';
      video.poster = example.poster;
      video.setAttribute('aria-label', example.title + ' video');
      var source = document.createElement('source');
      source.src = example.src;
      source.type = 'video/mp4';
      video.appendChild(source);
      var previous = slider.querySelector('video');
      if (previous) {
        previous.pause();
        previous.replaceChildren();
        previous.load();
      }
      slider.querySelector('.video-slot').replaceChildren(video);
      var position = (current + 1) + ' of ' + examples.length;
      var counter = slider.getAttribute('data-counter-style') === 'task'
        ? example.title + ' (' + position + ')'
        : 'Example ' + position;
      slider.querySelector('.example-counter').textContent = counter;
      slider.querySelector('.example-slide').setAttribute('aria-label', counter);
    });
  });
})();
