var posts=["2025/02/07/YatCPU-Lab-Notes/","2025/02/07/hello-world/","2025/02/08/OS-learning/"];function toRandomPost(){
    pjax.loadUrl('/'+posts[Math.floor(Math.random() * posts.length)]);
  };