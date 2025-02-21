var posts=["2025/02/19/Rust-String/","2025/02/08/OS-learning/","2025/02/19/Rust-iter/","2025/02/10/Rust-Notes/","2025/02/19/Rust-smart-pointer/","2025/02/07/hello-world/","2025/02/19/Rust-macro/","2025/02/19/Rust-包-Crate/","2025/02/07/YatCPU-Lab-Notes/"];function toRandomPost(){
    pjax.loadUrl('/'+posts[Math.floor(Math.random() * posts.length)]);
  };