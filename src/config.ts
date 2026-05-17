export const SITE = {
  website: "https://blog.javlim.dev/", // replace with your deployed domain
  author: "Javier Lim",
  profile: "https://javlim.dev/",
  desc: "Notes on building, shipping, and learning — by Javier Lim, entrepreneur and software engineer.",
  title: "Javier Lim",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Singapore", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
