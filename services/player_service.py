import base64
import os

def file_to_base64(file_path, mime_type):
    """Reads a file and returns its base64 data URI string."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def get_netflix_player_html(videos_dict, subtitles_dict, default_audio="English Dub"):
    """
    Generates HTML for a Video.js player with a consolidated Netflix-style menu for audio and subtitles,
    along with 10-second seek backward and forward buttons.
    videos_dict format: {"Language Name": "filepath.mp4"}
    subtitles_dict format: {"Language Name": "filepath.vtt"}
    """
    # 1. Convert files to Data URIs (Base64)
    video_uris = {}  # type: dict[str, str]
    for lang, path in videos_dict.items():
        video_uris[lang] = file_to_base64(path, "video/mp4")
        
    subtitle_uris = {}  # type: dict[str, str]
    for lang, path in subtitles_dict.items():
        subtitle_uris[lang] = file_to_base64(path, "text/vtt")
        
    # Generate JavaScript objects for tracks
    videos_js = "{\n"
    for lang, uri in video_uris.items():
        videos_js += f'        "{lang}": "{uri}",\n'
    videos_js += "    }"
    
    # 2. Build the HTML template
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <link href="https://vjs.zencdn.net/8.10.0/video-js.css" rel="stylesheet" />
  <link href="https://unpkg.com/@videojs/themes@1/dist/city/index.css" rel="stylesheet">
  <style>
    body {{
      margin: 0;
      padding: 0;
      background-color: #000;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }}
    .video-js {{
      width: 100%;
      height: 100%;
    }}
    /* Consolidated Menu Container */
    .vjs-netflix-menu {{
      position: absolute;
      bottom: 4em;
      right: 20px;
      background: rgba(0, 0, 0, 0.85);
      border-radius: 4px;
      padding: 20px;
      display: none;
      color: white;
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      z-index: 1001;
      min-width: 400px;
      backdrop-filter: blur(10px);
      box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}
    .vjs-netflix-menu.show {{
      display: flex;
    }}
    .menu-section {{
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 0 15px;
    }}
    .menu-section:not(:last-child) {{
      border-right: 1px solid rgba(255, 255, 255, 0.2);
    }}
    .menu-header {{
      font-size: 14px;
      font-weight: bold;
      color: rgba(255, 255, 255, 0.5);
      margin-bottom: 10px;
      text-transform: uppercase;
    }}
    .menu-item {{
      padding: 8px 0;
      cursor: pointer;
      font-size: 16px;
      transition: color 0.2s;
      display: flex;
      align-items: center;
    }}
    .menu-item:hover {{
      color: #e50914; /* Netflix Red */
    }}
    .menu-item.active {{
      font-weight: bold;
    }}
    .menu-item.active::before {{
      content: '✓';
      margin-right: 8px;
    }}
    /* Custom button styling */
    .vjs-netflix-btn, .vjs-seek-btn {{
      cursor: pointer;
      font-size: 1.5em;
    }}
    /* Subtitle positioning */
    .video-js .vjs-text-track-display {{
      bottom: 4em;
    }}
    
    /* Netflix layout specifics - Order of elements */
    .vjs-play-toggle {{
        order: 0;
    }}
    .vjs-seek-back-btn {{
        order: 1;
    }}
    .vjs-seek-forward-btn {{
        order: 2;
    }}
    .vjs-volume-panel {{
        order: 3;
    }}
    /* Push rest to right */
    .vjs-spacer {{
        order: 4;
    }}
    .vjs-remaining-time-display {{
        order: 5;
    }}
    .vjs-netflix-btn {{
        order: 6;
    }}
    .vjs-fullscreen-toggle {{
        order: 7;
    }}
  </style>
</head>
<body>
  <div style="position: relative; width: 100%; height: 100%;">
    <video
      id="netflix-player"
      class="video-js vjs-theme-city"
      controls
      preload="auto"
      autoplay
    >
      <source src="{video_uris.get(default_audio, list(video_uris.values())[0])}" type="video/mp4" />
      """
      
    # Add subtitle tracks
    # Dynamic lang-code mapping for subtitle tracks
    _SUBTITLE_CODE_MAP = {
        "English": "en", "Hindi": "hi", "Kannada": "kn",
        "en": "en", "hi": "hi", "kn": "kn",
    }
    for lang, uri in subtitle_uris.items():
        lang_code = _SUBTITLE_CODE_MAP.get(lang, lang[:2].lower())
        html_content += f"""
      <track kind="captions" src="{uri}" srclang="{lang_code}" label="{lang}" />"""
        
    html_content += f"""
    </video>
    
    <!-- Consolidated Netflix-Style Menu -->
    <div id="netflix-menu" class="vjs-netflix-menu">
      <div class="menu-section">
        <div class="menu-header">Audio</div>
        <div id="audio-list">
    """
    
    for lang in video_uris.keys():
        active = "active" if lang == default_audio else ""
        html_content += f"""
          <div class="menu-item {active}" data-type="audio" data-value="{lang}">{lang}</div>"""
          
    html_content += """
        </div>
      </div>
      <div class="menu-section">
        <div class="menu-header">Subtitles</div>
        <div id="subtitle-list">
          <div class="menu-item active" data-type="subtitle" data-value="off">Off</div>
    """
    
    for lang in subtitle_uris.keys():
        html_content += f"""
          <div class="menu-item" data-type="subtitle" data-value="{lang}">{lang}</div>"""
          
    html_content += """
        </div>
      </div>
    </div>
  </div>

  <script src="https://vjs.zencdn.net/8.10.0/video.min.js"></script>
  <script>
    // Initialize Video.js without default subtitle/audio buttons
    var player = videojs('netflix-player', {
        controlBar: {
            subsCapsButton: false,
            audioTrackButton: false,
            pictureInPictureToggle: false
        }
    });
    
    var videoSources = """ + videos_js + """;
    
    player.ready(function() {
      var controlBar = player.controlBar;
      
      // Add Seek Backward Button
      var backBtn = controlBar.addChild('button', {
        className: 'vjs-seek-back-btn vjs-seek-btn vjs-control vjs-button'
      });
      backBtn.el().innerHTML = '<span aria-hidden="true" class="vjs-icon-placeholder">⏪</span><span class="vjs-control-text">Back 10s</span>';
      backBtn.el().onclick = function(e) {
          e.stopPropagation();
          var newTime = Math.max(0, player.currentTime() - 10);
          player.currentTime(newTime);
      };

      // Add Seek Forward Button
      var fwdBtn = controlBar.addChild('button', {
        className: 'vjs-seek-forward-btn vjs-seek-btn vjs-control vjs-button'
      });
      fwdBtn.el().innerHTML = '<span aria-hidden="true" class="vjs-icon-placeholder">⏩</span><span class="vjs-control-text">Forward 10s</span>';
      fwdBtn.el().onclick = function(e) {
          e.stopPropagation();
          var newTime = Math.min(player.duration(), player.currentTime() + 10);
          player.currentTime(newTime);
      };
      
      // Add custom consolidated button for Audio and Subtitles
      var btn = controlBar.addChild('button', {
        className: 'vjs-netflix-btn vjs-control vjs-button'
      });
      btn.el().innerHTML = '<span aria-hidden="true" class="vjs-icon-placeholder">💬</span><span class="vjs-control-text">Audio and Subtitles</span>';
      
      var menu = document.getElementById('netflix-menu');
      
      btn.el().onclick = function(e) {
        e.stopPropagation();
        menu.classList.toggle('show');
      };
      
      document.addEventListener('click', function(e) {
        if (!menu.contains(e.target) && !btn.el().contains(e.target)) {
          menu.classList.remove('show');
        }
      });
      
      // Handle selections
      var items = document.querySelectorAll('.menu-item');
      items.forEach(function(item) {
        item.onclick = function() {
          var type = this.getAttribute('data-type');
          var val = this.getAttribute('data-value');
          
          // Manage active state in UI
          var siblings = this.parentNode.querySelectorAll('.menu-item');
          siblings.forEach(s => s.classList.remove('active'));
          this.classList.add('active');
          
          if (type === 'audio') {
            var currentTime = player.currentTime();
            var isPlaying = !player.paused();
            player.src({type: 'video/mp4', src: videoSources[val]});
            player.one('loadedmetadata', function() {
              player.currentTime(currentTime);
              if (isPlaying) player.play();
            });
          } else {
            // Handle Subtitles
            var tracks = player.textTracks();
            for (var i = 0; i < tracks.length; i++) {
              if (val === 'off') {
                tracks[i].mode = 'disabled';
              } else {
                tracks[i].mode = (tracks[i].label === val) ? 'showing' : 'disabled';
              }
            }
          }
        };
      });
    });
  </script>
</body>
</html>
    """
    return html_content
