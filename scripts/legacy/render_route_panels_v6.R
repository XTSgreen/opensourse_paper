# Real-data mini-panels for the editable draw.io roadmap.
# Inherit the existing v5 Arial typography and semantic colour system.
# Render at their actual placement size, using native Cairo text for PNGs.
suppressPackageStartupMessages(library(ggplot2))
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) normalizePath(args[1], winslash = "/") else getwd()
route <- if (length(args) >= 2) args[2] else file.path(root, "paper/technical_route_v6")
src <- file.path(route, "source_data")
out <- file.path(route, "R_panels")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
read <- function(name) read.csv(file.path(src, name), stringsAsFactors = FALSE)
blue <- "#2166AC"; red <- "#B2182B"; green <- "#1B7837"
purple <- "#762A83"; orange <- "#F1A340"; grey <- "#777777"
states_col <- c("#2166AC", "#92C5DE", "#1B7837", "#A6DBA0", "#762A83",
                "#C2A5CF", "#B2182B", "#EF8A62", "#F1A340", "#999999")
names(states_col) <- as.character(0:9)
theme_set(theme_classic(base_size = 5.5, base_family = "Arial") +
  theme(axis.line = element_line(linewidth = .22, colour = "#333333"),
        axis.ticks = element_line(linewidth = .2), axis.ticks.length = grid::unit(.65, "mm"),
        axis.text = element_text(size = 5.2, colour = "#222222"),
        axis.title = element_text(size = 5.7),
        axis.title.x = element_text(margin = margin(t = 1.2)),
        axis.title.y = element_text(margin = margin(r = 1.5)),
        legend.position = "bottom", legend.text = element_text(size = 5.2),
        legend.title = element_text(size = 5.2),
        legend.key.height = grid::unit(2.1, "mm"),
        legend.key.width = grid::unit(2.1, "mm"),
        legend.spacing.x = grid::unit(.5, "mm"),
        legend.margin = margin(0, 0, 0, 0), legend.box.spacing = grid::unit(.7, "mm"),
        plot.margin = margin(1.3, 1.7, 1.2, 1.4),
        panel.grid = element_blank()))

# Canvas is 1,800 draw.io units wide at 183 mm.
save_panel <- function(name, plot, width, height) {
  w <- width * 183 / 1800 / 25.4
  h <- height * 183 / 1800 / 25.4
  grDevices::svg(file.path(out, paste0(name, ".svg")), width=w, height=h,
                 pointsize=5.5, family="Arial", bg="white")
  print(plot); invisible(dev.off())
  cairo_pdf(file.path(out, paste0(name, ".pdf")), width=w, height=h,
            pointsize=5.5, family="Arial", bg="white")
  print(plot); invisible(dev.off())
  png(file.path(out, paste0(name, ".png")), width=w, height=h, units="in", res=600,
      type="cairo", bg="white")
  print(plot); invisible(dev.off())
  cat(name, sprintf("%.2f × %.2f mm\n", w*25.4, h*25.4))
}

cells <- read("GSE228154_cells.csv")
cells$state <- factor(cells$state, levels=0:9)
cells$timepoint <- factor(cells$timepoint, levels=c("D0", "D3", "D6", "D9"))
x <- as(Matrix::readMM(gzfile(file.path(src, "GSE228154_log_expression.mtx.gz"))), "CsparseMatrix")
set.seed(20260905)
pca <- irlba::prcomp_irlba(x, n=2, center=TRUE, scale.=FALSE)
variance <- sum(Matrix::colSums(x*x)/(nrow(x)-1) -
                (Matrix::colSums(x)^2)/(nrow(x)*(nrow(x)-1)))
explained <- 100*pca$sdev^2/variance
cells$PC1 <- pca$x[,1]; cells$PC2 <- pca$x[,2]
write.csv(cells, file.path(src, "GSE228154_R_PCA_coordinates.csv"), row.names=FALSE)
p <- ggplot(cells, aes(PC1, PC2, colour=state)) +
  geom_point(size=.36, alpha=.5, stroke=0) + scale_colour_manual(values=states_col) +
  labs(x=sprintf("PC1 (%.1f%%)", explained[1]), y=sprintf("PC2 (%.1f%%)", explained[2])) +
  theme(legend.position="none")
save_panel("01_expression_pca", p, 310, 240)

composition <- as.data.frame(table(cells$timepoint, cells$state))
names(composition) <- c("time", "state", "n")
composition$fraction <- composition$n / ave(composition$n, composition$time, FUN=sum)
write.csv(composition, file.path(src, "GSE228154_state_composition.csv"), row.names=FALSE)
p <- ggplot(composition, aes(time, fraction, fill=state)) + geom_col(width=.68) +
  scale_fill_manual(values=states_col, name=NULL, drop=FALSE) +
  scale_y_continuous(breaks=c(0,.5,1), labels=c("0", "50", "100"), expand=c(0,0)) +
  labs(x=NULL, y="Cells (%)") +
  guides(fill=guide_legend(nrow=2, byrow=TRUE))
save_panel("02_state_composition", p, 310, 240)

programmes <- read("GSE228154_state_programmes.csv")
mat <- as.matrix(programmes[,-1]); mat <- scale(mat)
heat <- data.frame(state=rep(0:9, 5), programme=rep(colnames(mat), each=10), z=as.vector(mat))
heat$programme <- factor(heat$programme, levels=rev(colnames(mat)),
                         labels=rev(c("EPI", "PROL", "EMT", "STEM", "SEN")))
p <- ggplot(heat, aes(state, programme, fill=z)) + geom_tile(linewidth=.12, colour="white") +
  scale_fill_gradient2(low=blue, mid="white", high=red, name="Programme z score",
                       limits=c(-2.5,2.5), breaks=c(-2,0,2)) +
  scale_x_continuous(breaks=0:9, expand=c(0,0)) + labs(x="State", y=NULL) +
  theme(axis.line=element_blank(), axis.ticks=element_blank()) +
  guides(fill=guide_colourbar(title.position="top", barwidth=grid::unit(17,"mm"),
                             barheight=grid::unit(1.4,"mm")))
save_panel("03_state_programmes", p, 310, 240)

curvature <- read("Fig2a_curvature.csv")
curvature$feature <- factor(curvature$feature, levels=rev(curvature$feature),
                           labels=rev(c("Prolif.", "EMT", "Stemness", "Senesc.", "Epithelial")))
p <- ggplot(curvature, aes(y=feature)) +
  geom_segment(aes(x=log10(balanced_ot), xend=log10(semi_relaxed_iot), yend=feature),
               colour="#C5C5C5", linewidth=.65) +
  geom_point(aes(x=log10(balanced_ot), colour="Balanced", shape="Balanced"), size=1.35, stroke=.4) +
  geom_point(aes(x=log10(semi_relaxed_iot), colour="UOT-IOT", shape="UOT-IOT"), size=1.4) +
  scale_colour_manual(values=c(Balanced=grey, "UOT-IOT"=blue), name=NULL) +
  scale_shape_manual(values=c(Balanced=1,"UOT-IOT"=16), name=NULL) +
  scale_x_continuous(limits=c(-12.5,.2), breaks=c(-12,-6,0)) +
  labs(x=expression(log[10]*" curvature"), y=NULL)
save_panel("04_curvature", p, 380, 200)

recovery <- read("Fig2b_recovery.csv")
recovery$pure_column_direction <- as.logical(recovery$pure_column_direction)
p <- ggplot(recovery, aes(x=truth)) + geom_abline(slope=1,intercept=0,colour="#B8B8B8",linetype=2,linewidth=.25) +
  geom_segment(aes(y=balanced_ot, xend=truth, yend=semi_relaxed_iot),colour="#BEBEBE",linewidth=.25) +
  geom_point(aes(y=balanced_ot, colour="Balanced", shape="Balanced"),size=1.25,stroke=.4) +
  geom_point(aes(y=semi_relaxed_iot, colour="UOT-IOT", shape="UOT-IOT"),size=1.3) +
  geom_point(data=subset(recovery,pure_column_direction), aes(y=semi_relaxed_iot),
             shape=1,colour=red,size=2.05,stroke=.35) +
  scale_colour_manual(values=c(Balanced=grey,"UOT-IOT"=blue),name=NULL) +
  scale_shape_manual(values=c(Balanced=1,"UOT-IOT"=16),name=NULL) +
  coord_fixed(xlim=c(-1,1.1),ylim=c(-1,1.1)) +
  scale_x_continuous(breaks=c(-1,0,1)) + scale_y_continuous(breaks=c(-1,0,1)) +
  labs(x="True coefficient",y="Recovered")
save_panel("05_synthetic_recovery", p, 380, 200)

sensitivity <- read("Fig2c_sensitivity.csv")
p <- ggplot(sensitivity,aes(mu,direction_sensitivity_norm)) +
  annotate("rect",xmin=.1,xmax=1,ymin=-Inf,ymax=Inf,fill=blue,alpha=.07) +
  geom_line(colour=blue,linewidth=.55) + geom_point(colour=blue,size=1.2) +
  geom_point(data=subset(sensitivity,mu==.5),shape=8,colour=red,size=2) +
  annotate("text",x=2.7,y=.67,label="paste('Working point: ',mu,' = 0.5')",parse=TRUE,
           size=5.2/2.8453,family="Arial") +
  scale_x_continuous(breaks=c(0,1,2,5)) + scale_y_continuous(breaks=c(.2,.4,.6,.8),limits=c(.16,.76)) +
  labs(x=expression("KL-anchor strength "*mu),y="Sensitivity norm")
save_panel("06_anchor_sensitivity", p, 380, 200)

external <- read("Fig2d_external_validation.csv")
external$site <- factor(external$site,levels=rev(external$site))
p <- ggplot(external,aes(gain_mean,site,colour=cohort)) +
  geom_segment(aes(x=ci_low,xend=ci_high,yend=site),linewidth=.55) + geom_point(size=1.55) +
  scale_colour_manual(values=c(GSE246662=blue,GSE183904=orange),name=NULL) +
  scale_x_continuous(breaks=c(.09,.11,.13),limits=c(.087,.132)) +
  labs(x="MAE reduction (95% CI)",y=NULL)
save_panel("07_external_transfer", p, 380, 200)

lineage <- read("Fig3b_lineage_paired_values.csv")
long <- rbind(data.frame(mouse=lineage$mouse,system=lineage$system,history="Epithelial",score=lineage$epithelial_lineage),
              data.frame(mouse=lineage$mouse,system=lineage$system,history="ever-EMT",score=lineage$ever_emt_lineage))
long$history <- factor(long$history,levels=c("Epithelial","ever-EMT"))
p <- ggplot(long,aes(history,score,group=mouse,colour=system,shape=system)) +
  geom_line(linewidth=.42,alpha=.7) + geom_point(size=1.15) +
  stat_summary(aes(group=system),fun=mean,geom="point",size=2.25,shape=18) +
  scale_colour_manual(values=c(TENA=blue,Cdh=red),labels=c("Cdh (n=3)","TENA (n=5)"),name=NULL) +
  scale_shape_manual(values=c(TENA=16,Cdh=15),labels=c("Cdh (n=3)","TENA (n=5)"),name=NULL) +
  scale_x_discrete(expand=expansion(add=.3)) + labs(x="Lineage history",y="Invasive-EMT score")
save_panel("08_lineage_pairs", p, 380, 255)

patients <- read("Fig4ab_patient_scores.csv")
patients$outcome <- factor(patients$outcome,levels=c("RD","pCR"))
ci <- subset(read("Fig4ab_descriptive_mean_ci.csv"), feature=="score_IFN_HLA")
p <- ggplot(patients,aes(outcome,score_IFN_HLA,colour=outcome,fill=outcome)) +
  geom_violin(width=.82,alpha=.15,linewidth=.2,trim=TRUE) +
  geom_point(position=position_jitter(width=.13,height=0,seed=20260905),size=.8,alpha=.65,stroke=0) +
  geom_errorbar(data=ci,aes(x=outcome,y=mean,ymin=mean_ci_low,ymax=mean_ci_high),
                inherit.aes=FALSE,width=.13,linewidth=.45,colour="#222222") +
  geom_point(data=ci,aes(x=outcome,y=mean),inherit.aes=FALSE,shape=23,size=1.8,fill="white",stroke=.55) +
  scale_colour_manual(values=c(RD=grey,pCR=orange)) + scale_fill_manual(values=c(RD=grey,pCR=orange)) +
  scale_x_discrete(labels=c("RD\nn=34","pCR\nn=45")) +
  labs(x=NULL,y="IFN/HLA score") + theme(legend.position="none")
save_panel("09_patient_response", p, 380, 255)

branches <- c("EPI","INV_EMT","INF_EMT","IFN_HLA","PROLIF","DORM_STRESS")
patients <- patients[order(patients$outcome,patients$score_IFN_HLA),]
z <- scale(as.matrix(patients[,paste0("score_",branches)]))
heat <- data.frame(patient=rep(seq_len(nrow(z)),length(branches)),
                   branch=rep(branches,each=nrow(z)),z=as.vector(z))
heat$branch <- factor(heat$branch,levels=rev(branches),
                      labels=rev(c("EPI","INV EMT","INF EMT","IFN/HLA","PROL","DORM")))
heat$z <- pmax(-3,pmin(3,heat$z))
write.csv(data.frame(donor_id=patients$donor_id,outcome=patients$outcome,z),
          file.path(src,"ARTEMIS_R_heatmap_z_scores.csv"),row.names=FALSE)
p <- ggplot(heat,aes(patient,branch,fill=z)) + geom_tile() +
  geom_vline(xintercept=34.5,colour="white",linewidth=.8) +
  scale_fill_gradient2(low=blue,mid="white",high=red,limits=c(-3,3),breaks=c(-3,0,3),
                       name="Branch z score",labels=c("<= -3","0",">= 3")) +
  scale_x_continuous(breaks=c(17.5,57),labels=c("RD (34)","pCR (45)"),expand=c(0,0)) +
  labs(x=NULL,y=NULL) + theme(axis.line=element_blank(),axis.ticks=element_blank(),
     legend.position="right",legend.title=element_text(size=5.2),legend.text=element_text(size=5.2)) +
  guides(fill=guide_colourbar(title.position="top",barheight=grid::unit(8,"mm"),
                             barwidth=grid::unit(1.6,"mm")))
save_panel("10_patient_programme_heatmap", p, 790, 150)

windows <- read("Fig5c_external_all_windows.csv")
means <- read("Fig5c_external_condition_state_summary.csv")
p <- ggplot(windows,aes(predicted,observed,colour=condition,shape=condition)) +
  geom_abline(slope=1,intercept=0,linetype=2,colour="#AAAAAA",linewidth=.3) +
  geom_point(size=.95,alpha=.45,stroke=0) +
  geom_point(data=means,size=2.2,alpha=1,stroke=.55,aes(fill=condition),colour="#222222") +
  scale_colour_manual(values=c(DMSO=blue,ispinesib="#D6604D"),name=NULL) +
  scale_fill_manual(values=c(DMSO=blue,ispinesib="#D6604D"),name=NULL,guide="none") +
  scale_shape_manual(values=c(DMSO=21,ispinesib=22),name=NULL) +
  coord_fixed(xlim=c(0,.9),ylim=c(0,.9)) +
  scale_x_continuous(breaks=c(0,.4,.8)) + scale_y_continuous(breaks=c(0,.4,.8)) +
  labs(x="Predicted composition",y="Observed composition")
# Shapes 21/22 need fill for window observations as well.
p$layers[[2]]$mapping <- aes(fill=condition)
save_panel("11_external_calibration", p, 430, 335)

metrics <- read("Fig5ab_state_metrics.csv")
metrics$partition <- factor(metrics$partition,levels=c("locked_external_e1","development_outer_oof"),
                            labels=c("External E1","Dev OOF"))
metrics$label <- ifelse(metrics$method=="PERSIST-IOT","PERSIST-IOT","No transport")
p <- ggplot(metrics,aes(state_cross_entropy,partition)) +
  geom_line(aes(group=partition),colour="#C5C5C5",linewidth=.7,orientation="y") +
  geom_point(aes(colour=label),size=1.8) +
  scale_colour_manual(values=c("PERSIST-IOT"=purple,"No transport"=grey),name=NULL) +
  scale_x_continuous(breaks=c(0,2,4),limits=c(0,5.3)) +
  labs(x="State cross-entropy",y=NULL)
save_panel("12_state_performance", p, 420, 170)

brier <- read("Fig5d_brier_paired_bootstrap.csv")
brier$partition <- factor(brier$partition,levels=c("locked_external_e1","development_outer_oof"),
                          labels=c("External E1","Dev OOF"))
p <- ggplot(brier,aes(difference_persist_minus_supervised,partition)) +
  geom_vline(xintercept=0,linetype=2,colour="#A0A0A0",linewidth=.3) +
  geom_segment(aes(x=ci_low,xend=ci_high,yend=partition),colour=purple,linewidth=.6) +
  geom_point(colour=purple,size=1.8) +
  scale_x_continuous(breaks=c(-.002,0,.004,.008),labels=c("-0.002","0","0.004","0.008")) +
  labs(x=expression(Delta*" Brier (95% CI)"),y=NULL)
save_panel("13_detection_brier", p, 420, 170)
cat("Rendered 13 mini-panels; all cell, mouse, patient and window records retained.\n")
