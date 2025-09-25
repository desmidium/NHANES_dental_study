library(pvclust) 
library(fpc)      
library(parallel) 

set.seed(42)

mydata <- read.csv("/data/complete/dataseet.csv")
mydata_scaled <- scale(mydata)  


evaluate_ch_index <- function(fit, data, k, dist_method) {
  data_num <- as.matrix(t(data))
  clust_labels <- cutree(as.hclust(fit$hclust), k = k)
  dist_mat <- dist(data_num, method = dist_method)
  stats <- cluster.stats(as.matrix(dist_mat), clust_labels)
  ch <- if (is.null(stats$ch)) NA else stats$ch
  return(data.frame(k = k, ch = ch))
}


fit_canberra <- pvclust(mydata, method.hclust = "complete", method.dist = "canberra", nboot = 1000, parallel = TRUE)
clusters_can <- pvpick(fit_canberra, alpha = 0.95)
output_file <- "/pathtoresults/pvpick.txt"
writeLines(unlist(lapply(clusters_can$clusters, paste, collapse = " ")), output_file)
png("/pathtoresults/pvclust.png", width = 1200, height = 600)
plot(fit_canberra, ylab = "Maximum Distance Between Clusters (Canberra)", cex.pv = 0.3, cex = 0.3, cex.axis = 0.3, cex.lab = 0.3, cex.sub = 0.3, cex.main = 0.3)
pvrect(fit_canberra, alpha = 0.95)
dev.off()

fit_canberra_sin <- pvclust(mydata, method.hclust = "single", method.dist = "canberra", nboot = 1000, parallel = TRUE)
fit_canberra_ave <- pvclust(mydata, method.hclust = "average", method.dist = "canberra", nboot = 1000, parallel = TRUE)
fit_euclidean <- pvclust(mydata_scaled, method.hclust = "complete", method.dist = "euclidean", nboot = 1000, parallel = TRUE)

max_k <- min(50, ncol(mydata))

results_canberra <- do.call(rbind, lapply(2:max_k, function(k) evaluate_ch_index(fit_canberra, mydata, k, "canberra")))
results_canberra_sin <- do.call(rbind, lapply(2:max_k, function(k) evaluate_ch_index(fit_canberra_sin, mydata, k, "canberra")))
results_canberra_ave <- do.call(rbind, lapply(2:max_k, function(k) evaluate_ch_index(fit_canberra_ave, mydata, k, "canberra")))
results_euclidean <- do.call(rbind, lapply(2:max_k, function(k) evaluate_ch_index(fit_euclidean, mydata_scaled, k, "euclidean")))

results_canberra$method <- "Canberra - Complete"
results_canberra_sin$method <- "Canberra - Single"
results_canberra_ave$method <- "Canberra - Average"
results_euclidean$method <- "Euclidean - Complete"

ch_results <- rbind(
  results_canberra,
  results_canberra_sin,
  results_canberra_ave,
  results_euclidean
)

ch_results <- ch_results[, c("method", "k", "ch")]

write.csv(ch_results, "/path/ch_index.csv", row.names = FALSE)

print(head(ch_results))