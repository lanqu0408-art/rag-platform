# HanLP Local Assets

This project stores HanLP runtime assets under:

- `models/hanlp_home/`

Code sets:

- `HANLP_HOME=<repo_root>/models/hanlp_home`
- `HANLP_URL=https://file.hankcs.com/hanlp/` (default fallback)

To refresh assets from local machine cache:

```bash
rsync -a --delete ~/.hanlp/ ./models/hanlp_home/
```
