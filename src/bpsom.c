#define VERSION "PD1.00"

/*
   ARCHITECTURE: A combination of a multilayered feedforword
   network (MFN) and one or more self-organising maps (SOMs); each
   hidden layer of the MFN can have his corresponding SOM.

   LEARNING IN THE MFN: A combination of supervised learning
   with the traditional back-propagation (BP) learning rule
   guided by clustering information in the SOMs.

   PUBLICATIONS about BP-SOM:
   Weijters, A. (1995). The BP-SOM architecture and learning algorithm.
      Neural Processing Letters, Vol 2:6, 13 - 16.
   Weijters, A. and van den Bosch, A. and van den Herik, H.J. (1977).
      Behavioural Aspects of BP-SOM (Accepted for publication in
      Connection Science).
   Ton Weijters, H. Jaap van den Herik, Antal van den Bosch, and
     Eric Postma (1977). Avoiding Overfitting with BP-SOM. Accepted
     for publication in the proceedings of the IJCAI97.
*/

#include <stdio.h>
#include <stddef.h>
/* #include <malloc.h> */
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#define INCLUDE1 "bp.h"
#define INCLUDE2 "som.h"
#define INCLUDE3 "grapho.hlp"

#define EVER ;;
#define MAX_UNITS 500
#define MAX_LAST_LAYER 5
#define NAME_L 128
#define DONOTHING 0.0001
#define SOM_LR_FALL 1
#define SOM_CONTEXT_FALL 4
#define MIN_IMPROVE 0.005
#define PATWIDTH 9

#define M_DIM_SOM_NETWORK 200
#define MAX_LABELS 200
#define LENGTH_ALL_LABELS_STR 1024
#define LABELL 3             /* don't change (layout in log-file) */

/* BP-SOM functions  */
void  complete_epoch(int lrn_yn, int material);
void  administrate_complete_epoch(int lrn_yn, int material, float mse,
                                 float p_correct, long int tel_ok);
void  update_som_context_and_lr(void);
void  read_parameters(char *prm_file);
void  open_log(void);
void  dump_parameters(void);
void  cheque_parameters(void);
void  allocate_memory(void);
void  dump_time(void);
void  open_fpin(char *filename);
void  open_fpout(char *filename);
void  random_weights(void);
void  save_environment(char *filename);
void  read_environment(char *filename);
void  abstract(void);
void  reset_bp_confusion_matrix(void);
void  dump_bp_confusion_matrix(void);
void  make_labels_from_all_labels(void);
void  fatal_error(void);
void  drange_cheque(char *par, int *value,
                    int min, int max, int def);
void  frange_cheque(char *par, float *value,
                    float min, float max, float def);

/* global BP-SOM-variables  */

char  classlabel[MAX_LABELS+1][LABELL+1];

int   onact[PATWIDTH];

int   part_win_x[MAX_LAST_LAYER+1], part_win_y[MAX_LAST_LAYER+1],
      use_som_error[MAX_LAST_LAYER+1],
      n_pruned_units[MAX_LAST_LAYER+1],
      epoch=0, nothing_done=0, pruned=0, last_layer;
long  int example=0, som_error_use_counter[MAX_LAST_LAYER+1];

float fraction[MAX_LABELS+1];    /* remark: fraction[0]=total */

FILE *fpin, *fpout, *fplog;

/* abstract variables */

int   best_epoch=0,
      best_n_pruned_units[MAX_LAST_LAYER+1],
      bp_confusion_matrix[MAX_LABELS+1][MAX_LABELS+1];

float best_p_correct=0.0,
      best_class[4], end_class[4],
      best_mse[4], end_mse[4],
      best_som[MAX_LAST_LAYER+1][4], end_som[MAX_LAST_LAYER+1][4];

/* BP-variables */

int   start_momentum=0;

float *weight[MAX_LAST_LAYER][MAX_UNITS+1],
      *del_old[MAX_LAST_LAYER][MAX_UNITS+1],
      *act[MAX_LAST_LAYER+1],
      *error[MAX_LAST_LAYER+1],
      *avg_act[MAX_LAST_LAYER+1],
      *std_act[MAX_LAST_LAYER+1],
      *target;

/* SOM-variables */

int   *som_winner_percents[MAX_LAST_LAYER+1][M_DIM_SOM_NETWORK+1],
      som_cont, n_labels, labeln,
      winner_reliability[MAX_LAST_LAYER+1],
      part_winner_reliability[MAX_LAST_LAYER+1],
      winner_labeln[MAX_LAST_LAYER+1];

long  int *som_conf_matrix[MAX_LAST_LAYER+1][MAX_LABELS+1],
          *som_cell_teller[MAX_LAST_LAYER+1]
            [M_DIM_SOM_NETWORK+1][M_DIM_SOM_NETWORK+1],
          som_ok[MAX_LAST_LAYER+1];

float *som_network_vector[MAX_LAST_LAYER+1]
        [M_DIM_SOM_NETWORK+1][M_DIM_SOM_NETWORK+1],
      som_total_distance[MAX_LAST_LAYER+1], som_lr,
      part_winner_dis[MAX_LAST_LAYER+1];

/* BP-SOM-parameter */

float SOM_ERROR_USE;

/* BP parameters */

char  LRN[NAME_L], VAL[NAME_L], TST[NAME_L],
      LOG[NAME_L], OLD_ENV[NAME_L], BEST_ENV[NAME_L];

int   SEED, N[MAX_LAST_LAYER+1],
      DUMP, DUMP_LAYER[MAX_LAST_LAYER+1],
      N_EPOCHS, TEST_RATIO, STOP_IF_NOTHING_DONE,
      MM_AFTER_N_EXAMPLES;

float BP_LEARN_RATE, MOMENTUM, ERROR_LEARN_THRESHOLD,
      WEIGHT_DECAY_RATE, PRUNE_THRESHOLD;

/* SOM-parameters */

char  ALL_LABELS[LENGTH_ALL_LABELS_STR];

int   DIM_SOM[MAX_LAST_LAYER+1], SOM_MAX_CONTEXT,
      SOM_MIN_CONTEXT, RELIABILITY_THRESHOLD;

float SOM_MAX_LEARN_RATE, SOM_MIN_LEARN_RATE;


#include INCLUDE1
#include INCLUDE2
#include INCLUDE3

int main(argc, argv)
int argc;
char **argv;
{
  int i;
  char buff[NAME_L];
  /* setbuf(stdout, NULL); */
  if (argc>=2) {
    read_parameters(argv[1]);
  } else {
    printf("\nName parameter file: "); gets(buff); read_parameters(buff);
  }
  open_log();
  dump_parameters();
  cheque_parameters();
  make_labels_from_all_labels();
  if (strlen(OLD_ENV)>1) {
    read_environment(OLD_ENV);
  } else {
    allocate_memory();
    if (argc==3) {
      SEED=atoi(argv[2]);
      fprintf(fplog, "\nSEED=%d used!", SEED);
    }
    srand(SEED);
    random_weights();
    clear_som_cell_counters();
  }

  clear_prune_information();
  for (i=0;i<=MAX_LABELS;i++) { fraction[i]=0.0; }
  
  for (EVER) {
    epoch++; nothing_done++;
    if ((epoch > N_EPOCHS) || (nothing_done>STOP_IF_NOTHING_DONE)) {
      abstract();
      dump_time();
      fprintf(fplog, "\nBP_SOM (%s) ready\n\n", VERSION);
      fclose(fplog);
      exit(0);
    }
    update_som_context_and_lr();
    reset_avg_act(); reset_std_act();

    /* learn epoch */
    complete_epoch(1, 1);
    if ((epoch % TEST_RATIO)==0) {
      count_avg_act(example);
      reset_std_act();

      /* classlabeling epoch on LRN-material */
      clear_som_cell_counters();
      complete_epoch(0, 1);
      count_som_cell_winners();
      count_std_act(example);
      dump_som_labeling();
      dump_avg_std();

      /* validation epoch */
      complete_epoch(0, 2);
      if (PRUNE_THRESHOLD>DONOTHING) { prune_if_possible(); }

      /* test epoch */
      complete_epoch(0, 3);
    }
  }
}

void complete_epoch(int lrn_yn, int material)
{
  int l, vb_ok;
  long int tel_ok=0;
  float mse=0.0, p_correct=0.0, example_error=0.0;

  example=0;
  for (l=0;l<=last_layer;l++) {
    som_total_distance[l]=0.0; som_ok[l]=0; som_error_use_counter[l]=0;
  }

  if (material==1) { open_fpin(LRN); }
  if (material==2) { open_fpin(VAL); }
  if (material==3) { open_fpin(TST); }
  if (DUMP>0) clear_som_conf_matrix();

  for (EVER) {

    if (DUMP>2) {
      man_read_input_target();
    } else {
      read_input_target();
    }
    example++;
    if ((epoch==1) && (lrn_yn==1)) {
      fraction[0]++; fraction[labeln]++;
    }
    forward();
    if (res_ok()==1) { tel_ok++; vb_ok=1; } else { vb_ok=0; }
    mse+=(example_error=count_error());
    process_som_vectors(lrn_yn, material);
    
    if (DUMP>0) dump_result(vb_ok);
    if (DUMP>2) screen_dump_result(vb_ok);
    if (material==1) {
      if ((epoch % TEST_RATIO)==0) {
        if (lrn_yn==1) { update_avg_act(); } else { update_std_act(); }
      }
    }

    /* backward */
    if ((lrn_yn==1) && (example_error>ERROR_LEARN_THRESHOLD)) {
      if (start_momentum>=MM_AFTER_N_EXAMPLES) {
        backprop(MOMENTUM);
      } else {
        backprop(0.0);
        start_momentum++;
      }
    }

    if (feof(fpin)) {
      fclose(fpin);
      administrate_complete_epoch(lrn_yn, material,
                                  mse, p_correct, tel_ok);
      break;
    }
  }
}

void administrate_complete_epoch(int lrn_yn, int material, float mse,
float p_correct, long int tel_ok)
{
  int l, i;
  if (example>0) {
    /* after the first lrn-epoch, dump fraction information */
    if ((epoch==1) && (lrn_yn==1)) {
      fprintf(fplog, "\n\nClass fractions in the learning material:\n");
      for (i=1; i<=n_labels; i++) { fprintf(fplog, "%5d", i); }
      fprintf(fplog, "\n");
      for (i=1; i<=n_labels; i++) {
        fraction[i]=(fraction[i]/fraction[0]);
        fprintf(fplog, "%5.2f", fraction[i]);
      }
      fprintf(fplog, "\n");
    }

    /* save_environment("last.env"); */

    mse=(mse/example);
    p_correct=(100.0*tel_ok)/example;
    printf("\n%d/%d LRN=%d MAT=%d MSE=%5.3f CE=%6.2f USE",
      epoch, N_EPOCHS, lrn_yn, material, mse, 100.0-p_correct);
    fprintf(fplog, "\n%d/%d LRN=%d MAT=%d MSE=%5.3f CE=%6.2f USE",
      epoch, N_EPOCHS, lrn_yn, material, mse, 100.0-p_correct);
    for (l=1; l<=last_layer; l++) {
      if (DIM_SOM[l]>0) {
      /* som_total_distance[l]=(int) (som_total_distance[l]/example); */
      printf(" SOM%d=%5.1f ", l, (100.0*som_error_use_counter[l]/example));
      fprintf(fplog, " SOM%d=%5.1f", l, (100.0*som_error_use_counter[l]/example));
      }
    }

    /* update abstract information */
    end_class[material]=p_correct;
    end_mse[material]=mse;
    for (l=0; l<=last_layer; l++) {
      end_som[l][material]=100.0*som_ok[l]/example;
    }

    if (material==2) {
      if (((p_correct>best_p_correct) ||
          ((p_correct==best_p_correct) &&
           (mse+MIN_IMPROVE<best_mse[2]))) ||
          ((pruned==1) && (p_correct>=best_p_correct))) {
      best_p_correct=p_correct; best_mse[2]=mse;
      best_epoch=epoch; nothing_done=0; pruned=0;
        if (strlen(BEST_ENV)>1) {
          printf("*");
          fprintf(fplog, "*");
          save_environment(BEST_ENV);
        }
      }
    }

    if ((material==3) && (epoch==best_epoch)) {
      for (i=1; i<=3; i++) {
        best_class[i]=end_class[i];
        best_mse[i]=end_mse[i];
        for (l=0; l<=last_layer; l++) {
          best_som[l][i]=end_som[l][i];
        }
        for (l=1; l<last_layer; l++) {
          best_n_pruned_units[l]=n_pruned_units[l];
        }
      }
    }
    if ((material==1) && (lrn_yn==1) && (DUMP>0)) { dump_soms_epoch_score(); }
    if ((DUMP>0) && (material==3)) { dump_som_conf_matrix(); }
  }
}

void update_som_context_and_lr(void)
{
  /* update SOM-context */
  som_cont=SOM_MIN_CONTEXT+
    floor((pow((double) N_EPOCHS-epoch, (double) SOM_CONTEXT_FALL)/
           pow((double) N_EPOCHS, (double) SOM_CONTEXT_FALL)) *
             (SOM_MAX_CONTEXT-SOM_MIN_CONTEXT) + 0.5);

  /* update SOM-learning-rate */
  som_lr=SOM_MIN_LEARN_RATE+
    (float) ((pow((double) N_EPOCHS-epoch, (double) SOM_LR_FALL)/
              pow((double) N_EPOCHS, (double) SOM_LR_FALL)) *
                (SOM_MAX_LEARN_RATE-SOM_MIN_LEARN_RATE));
}

void read_parameters(char *prm_file)
{
  int i;
  if ((fpin=fopen(prm_file, "r"))==NULL) {
    printf("\nOpening parameterfile %s failed", prm_file);
    exit(0);
  }
  /* read bp parameters */
  fscanf(fpin, "DUMP = %d", &DUMP);
  fscanf(fpin, "\nDUMP_LAYER = %d", &DUMP_LAYER[0]);
  for (i=1; i<=MAX_LAST_LAYER; i++) { fscanf(fpin, "%d", &DUMP_LAYER[i]); }
  fscanf(fpin, "\nSEED = %d", &SEED);

  fscanf(fpin, "\nLRN = %s", LRN);
  fscanf(fpin, "\nVAL = %s", VAL);
  fscanf(fpin, "\nTST = %s", TST);
  fscanf(fpin, "\nLOG = %s", LOG);
  fscanf(fpin, "\nOLD_ENV = %s", OLD_ENV);
  fscanf(fpin, "\nBEST_ENV = %s", BEST_ENV);
  fscanf(fpin, "\nALL_LABELS = %s", ALL_LABELS);
  fscanf(fpin, "\nN_EPOCHS = %d", &N_EPOCHS);
  fscanf(fpin, "\nSTOP_IF_NOTHING_DONE = %d", &STOP_IF_NOTHING_DONE);
  fscanf(fpin, "\nTEST_RATIO = %d", &TEST_RATIO);
  fscanf(fpin, "\nSOM_ERROR_USE = %f", &SOM_ERROR_USE);
  fscanf(fpin, "\nRELIABILITY_THRESHOLD = %d", &RELIABILITY_THRESHOLD);
  fscanf(fpin, "\nPRUNE_THRESHOLD =%f", &PRUNE_THRESHOLD);

  fscanf(fpin, "\nBP-architecture = %d", &N[0]);
  for (i=1; i<=MAX_LAST_LAYER; i++) { fscanf(fpin, "%d", &N[i]); }
  fscanf(fpin, "\nBP_LEARN_RATE = %f", &BP_LEARN_RATE);
  fscanf(fpin, "\nMOMENTUM = %f", &MOMENTUM);
  fscanf(fpin, "\nMM_AFTER_N_EXAMPLES = %d", &MM_AFTER_N_EXAMPLES);
  fscanf(fpin, "\nERROR_LEARN_THRESHOLD = %f", &ERROR_LEARN_THRESHOLD);
  fscanf(fpin, "\nWEIGHT_DECAY_RATE = %f", &WEIGHT_DECAY_RATE);

  fscanf(fpin, "\nSOM-architecture = %d", &DIM_SOM[0]);
  for (i=1; i<=MAX_LAST_LAYER; i++) { fscanf(fpin, "%d", &DIM_SOM[i]); }
  fscanf(fpin, "\nSOM_MAX_CONTEXT = %d", &SOM_MAX_CONTEXT);
  fscanf(fpin, "\nSOM_MIN_CONTEXT = %d", &SOM_MIN_CONTEXT);
  fscanf(fpin, "\nALL_LABELS = %s", ALL_LABELS);
  fscanf(fpin, "\nSOM_MAX_LEARN_RATE = %f", &SOM_MAX_LEARN_RATE);
  fscanf(fpin, "\nSOM_MIN_LEARN_RATE = %f", &SOM_MIN_LEARN_RATE);
  fclose(fpin);
  if (strlen(LOG)<1) {
    strncpy(LOG, prm_file, strlen(prm_file)-4);
    strcat(LOG, ".log");
    printf("Warning: problems with log-file name (%s used)", LOG);
  }
}

void open_log(void)
{
  if ((fplog = fopen(LOG, "a")) == NULL) {
    printf("\nFatal error: open log file %s failed!", LOG);
    exit(0);
  }
  fprintf(fplog, "NEW SESSION BPSOM %s (Copyright 1997 ton weijters)",
                  VERSION); dump_time();
}

void dump_parameters(void)
{
  int i;
  cheque_parameters();
  fprintf(fplog, "\nDUMP = %d", DUMP);
  fprintf(fplog, "\nDUMP_LAYER = %d", DUMP_LAYER[0]);
  for (i=1; i<=MAX_LAST_LAYER; i++) { fprintf(fplog, " %d", DUMP_LAYER[i]); }
  fprintf(fplog, "\nSEED = %d", SEED);

  fprintf(fplog, "\n\nLRN = %s", LRN);
  fprintf(fplog, "\nVAL = %s", VAL);
  fprintf(fplog, "\nTST = %s", TST);
  fprintf(fplog, "\nLOG = %s", LOG);
  fprintf(fplog, "\nOLD_ENV = %s", OLD_ENV);
  fprintf(fplog, "\nBEST_ENV = %s", BEST_ENV);
  fprintf(fplog, "\nALL_LABELS = %s", ALL_LABELS);
  fprintf(fplog, "\nN_EPOCHS = %d", N_EPOCHS);
  fprintf(fplog, "\nSTOP_IF_NOTHING_DONE = %d", STOP_IF_NOTHING_DONE);
  fprintf(fplog, "\nTEST_RATIO = %d", TEST_RATIO);
  fprintf(fplog, "\nSOM_ERROR_USE = %f", SOM_ERROR_USE);
  fprintf(fplog, "\nRELIABILTY_THRESHOLD = %d", RELIABILITY_THRESHOLD);
  fprintf(fplog, "\nPRUNE_THRESHOLD = %f", PRUNE_THRESHOLD);

  fprintf(fplog, "\n\nBP-architecture = %d ", N[0]);
  for (i=1; i<=MAX_LAST_LAYER; i++) { fprintf(fplog, "%d ", N[i]); }
  fprintf(fplog, "\nBP_LEARN_RATE = %f", BP_LEARN_RATE);
  fprintf(fplog, "\nMOMENTUM = %f", MOMENTUM);
  fprintf(fplog, "\nMM_AFTER_N_EXAMPLES = %d", MM_AFTER_N_EXAMPLES);
  fprintf(fplog, "\nERROR_LEARN_THRESHOLD = %f", ERROR_LEARN_THRESHOLD);
  fprintf(fplog, "\nWEIGHT_DECAY_RATE = %f", WEIGHT_DECAY_RATE);

  fprintf(fplog, "\n\nSOM-architecture = %d", DIM_SOM[0]);
  for (i=1; i<=MAX_LAST_LAYER; i++) { fprintf(fplog, " %d", DIM_SOM[i]); }
  fprintf(fplog, "\nSOM_MAX_CONTEXT = %d", SOM_MAX_CONTEXT);
  fprintf(fplog, "\nSOM_MIN_CONTEXT = %d", SOM_MIN_CONTEXT);
  fprintf(fplog, "\nSOM_MAX_LEARN_RATE = %f", SOM_MAX_LEARN_RATE);
  fprintf(fplog, "\nSOM_MIN_LEARN_RATE = %f", SOM_MIN_LEARN_RATE);
  fclose(fplog);
  fopen(LOG, "a");
}

void cheque_parameters(void)
{
  int i;
  drange_cheque("DUMP", &DUMP, 0, 3, 0);
  for (i=0; i<=MAX_LAST_LAYER; i++) {
    drange_cheque("DUMPLAYER", &DUMP_LAYER[i], 0, 1, -1);
  }
  drange_cheque("SEED", &SEED, 0, 32000, 7);
  /*
  LRN LRN
  VAL VAL
  TST TST
  LOG LOG
  OLD_ENV OLD_ENV
  BEST_ENV BEST_ENV
  ALL_LABELS ALL_LABELS
  */
  drange_cheque("N_EPOCHS", &N_EPOCHS, 1, INT_MAX, 2000);
  drange_cheque("STOP_IF_NOTHING_DONE", &STOP_IF_NOTHING_DONE, TEST_RATIO, N_EPOCHS, 100);
  drange_cheque("TEST_RATIO", &TEST_RATIO, 1, N_EPOCHS, 5);
  frange_cheque("SOM_ERROR_USE", &SOM_ERROR_USE, 0.0, 1.0, 0.25);
  drange_cheque("RELIABILTY_THRESHOLD", &RELIABILITY_THRESHOLD, 0, 100, 95);
  frange_cheque("PRUNE_THRESHOLD", &PRUNE_THRESHOLD, 0.0, 1.0, 0.02);
  for (i=0; i<=MAX_LAST_LAYER; i++) {
     drange_cheque("BP-architecture", &N[i], 0, MAX_UNITS, -1);
  }
  frange_cheque("BP_LEARN_RATE", &BP_LEARN_RATE, 0.0, 1.0, 0.15);
  frange_cheque("MOMENTUM", &MOMENTUM, 0.0, 1.0, 0.40);
  drange_cheque("MM_AFTER_N_EXAMPLES", &MM_AFTER_N_EXAMPLES, 0, 32000, 100);
  frange_cheque("ERROR_LEARN_THRESHOLD", &ERROR_LEARN_THRESHOLD, 0.0, 1.0, 0.02);
  frange_cheque("WEIGHT_DECAY_RATE", &WEIGHT_DECAY_RATE, 0.0, 0.01, 0.00001);
  for (i=1; i<=MAX_LAST_LAYER; i++) {
    drange_cheque("SOM-architecture", &DIM_SOM[i], 0, M_DIM_SOM_NETWORK, -1);
  }
  drange_cheque("SOM_MAX_CONTEXT", &SOM_MAX_CONTEXT, SOM_MIN_CONTEXT, 10, 2);
  drange_cheque("SOM_MIN_CONTEXT", &SOM_MIN_CONTEXT, 0, SOM_MAX_CONTEXT, 0);
  frange_cheque("SOM_MAX_LEARN_RATE", &SOM_MAX_LEARN_RATE, SOM_MIN_LEARN_RATE, 1.0, 0.20);
  frange_cheque("SOM_MIN_LEARN_RATE", &SOM_MIN_LEARN_RATE, 0.0, SOM_MAX_LEARN_RATE, 0.05);

  n_labels = (int) ((strlen(ALL_LABELS)+1)/(LABELL+1));
  if (n_labels>MAX_LABELS) {
    printf("\nFatal error: to many labels %d (max = %d)",
            n_labels, MAX_LABELS);
    fatal_error();
  }
  for (i=0; i<=MAX_LAST_LAYER; i++) {
    if (N[i]>0) {
      last_layer=i;
    }
    if (N[i]>MAX_UNITS) {
      printf("\nMFN-layer %d = %d (max %d): Fatal error!",
               i, N[i], MAX_UNITS);
      fatal_error();
    }
  }
  if ((INCLUDE3=="proben1.hlp") && (n_labels!=N[last_layer])) {
    printf("\nWarning: posibble error in labelstring (%d) or dim-outputlayer MFN (%d)",
    n_labels, N[last_layer]);
  }

  for (i=0; i<=MAX_LAST_LAYER; i++) {
    if (DIM_SOM[i]>M_DIM_SOM_NETWORK) {
      printf("\nSOM-%d-dim %d (max %d): Fatal error!",
             i, DIM_SOM[i], M_DIM_SOM_NETWORK);
      fatal_error();
    }
  }
}

void allocate_memory(void)
{
  int l, i, x, y;
  for (l=0;l<last_layer;l++) {
    for (i=0;i<=N[l];i++) {
      weight[l][i]=(float *) calloc((unsigned) N[l+1]+1, sizeof(float));
      del_old[l][i]=(float *) calloc((unsigned) N[l+1]+1, sizeof(float));
    }
  }

  for (l=0;l<=last_layer;l++) {
    act[l]=(float *) calloc((unsigned) N[l]+1, sizeof(float));
    if (l>0) {
      error[l]=(float *) calloc((unsigned) N[l]+1, sizeof(float));
    }
    if ((l>0) && (l<last_layer)) {
      avg_act[l]=(float *) calloc((unsigned) N[l]+1, sizeof(float));
      std_act[l]=(float *) calloc((unsigned) N[l]+1, sizeof(float));
    }
  }

  for (l=0;l<=last_layer;l++) {
    for (x=1;x<=DIM_SOM[l];x++) {
      for (y=1;y<=DIM_SOM[l];y++) {
        som_network_vector[l][x][y] =
              (float *) calloc((unsigned) N[l]+1, sizeof(float));
        som_cell_teller[l][x][y] =
            (long int *) calloc((unsigned) (n_labels+1), sizeof(long int));
      }
    }
    for (i=1; i<=n_labels+1; i++) {
      som_conf_matrix[l][i] =
        (long int *) calloc((unsigned) (n_labels+1), sizeof(long int));
    }
    for (i=1; i<=DIM_SOM[l]; i++) {
      som_winner_percents[l][i] =
        (int *) calloc((unsigned) (DIM_SOM[l]+1), sizeof(int));
    }
  }
  target=(float *) calloc((unsigned) N[last_layer]+1, sizeof(float));
  if (target==NULL){
    printf("\nOut of memory. Allocation failed!");
    exit(0);
  }
}

void dump_time(void)
{
  struct tm *curtime;
  time_t bintime;
  time(&bintime);
  curtime=localtime(&bintime);
  fprintf(fplog, "\nTime: %s", asctime(curtime));
}

void open_fpin(char *filename)
{
  if ((fpin=fopen(filename, "r"))==NULL) {
    printf("\nImpossible to open %s!", filename);
    fatal_error();
  }
}

void open_fpout(char *filename)
{
  if ((fpout=fopen(filename, "w"))==NULL) {
    printf("\nImpossible to create %s!", filename);
    fatal_error();
  }
}

void random_weights(void)
{
  int l, i, j, x, y;
  float random_average=0.0;

  for (i=1; i<=1000; i++) {
    random_average+=rand()/1000;
  }

  for (l=0; l<last_layer; l++) {
    for (i=0; i<=N[l]; i++) {
      for (j=1; j<=N[l+1]; j++) {
        weight[l][i][j]=(float) ((random_average - rand())/
                                  random_average);
      }
    }
  }

  for (l=0; l<last_layer; l++) {
    for (x=1;x<=DIM_SOM[l];x++) {
      for (y=1;y<=DIM_SOM[l];y++) {
        for (i=1; i<=N[l]; i++) {
          som_network_vector[l][x][y][i] =
            (0.5 + (float) (random_average - rand())/
                           random_average);
        }
      }
    }
  }
}

void save_environment(char *filename)
{
  int l, i, j, x, y;
  open_fpout(filename);

  for (l=0; l<=MAX_LAST_LAYER; l++) { fprintf(fpout, "%d ", N[l]); }
  fprintf(fpout, "\n");

  for (l=0; l<=MAX_LAST_LAYER; l++) { fprintf(fpout, "%d ", DIM_SOM[l]); }
  fprintf(fpout, "\n");

  for (l=0; l<last_layer; l++) {
    for (i=0; i<=N[l]; i++) {
      for (j=1; j<=N[l+1]; j++) {
        fprintf(fpout, "%7.4f ", weight[l][i][j]);
      }
      fprintf(fpout, "\n");
    }
  }

  /* write SOM-labels, reliability and weights */
  for (l=0; l<=last_layer; l++) {
    for (x=1; x<=DIM_SOM[l]; x++) {
      for (y=1; y<=DIM_SOM[l]; y++) {
        /* label + reliability */
        fprintf(fpout, "%3ld %3d ",
          som_cell_teller[l][x][y][0],
          som_winner_percents[l][x][y]);
          /* vector */
          for (i=1; i<=N[l]; i++) {
     fprintf(fpout, "%7.4f ", som_network_vector[l][x][y][i]);
        }
        fprintf(fpout, "\n");
      }
    }
  }
  fclose(fpout);
}

void read_environment(char *filename)
{
  int l, i, j, x, y;

  fprintf(stderr,"opening %s\n",filename);
  
  open_fpin(filename);
  fprintf(fplog, "\n\nNew network format: ");
  for (l=0; l<=MAX_LAST_LAYER; l++) {
    fscanf(fpin, "%d", &N[l]);
    if (N[l]>0) { last_layer=l; }
    fprintf(fplog, "%d ", N[l]);
    if (N[l]>MAX_UNITS) {
      fprintf(fplog,
        "\nLayer = %d: Fatal error; impossible network architecture!", l);
      fatal_error();
    }
  }

  fprintf(fplog, "\nDIM_SOMS = ");
  for (l=0; l<=MAX_LAST_LAYER; l++) {
    fscanf(fpin, "%d ", &DIM_SOM[l]);
    fprintf(fplog, "%d ", DIM_SOM[l]);
  }

  if (epoch==0) { allocate_memory(); }

  /* read BP-weights */
  for (l=0; l<last_layer; l++) {
    for (i=0; i<=N[l]; i++) {
      for (j=1; j<=N[l+1]; j++) {
        fscanf(fpin, "%f ", &weight[l][i][j]);
      }
    }
  }

  /* read SOM-labels, reliabilty and weights */
  for (l=0; l<=last_layer; l++) {
    for (x=1; x<=DIM_SOM[l]; x++) {
      for (y=1; y<=DIM_SOM[l]; y++) {
        /* label + reliability */
        fscanf(fpin, "%ld %d ",
               &som_cell_teller[l][x][y][0],
               &som_winner_percents[l][x][y]);
        /* vector */
        for (i=1; i<=N[l]; i++) {
          fscanf(fpin, "%f ", &som_network_vector[l][x][y][i]);
        }
        fscanf(fpin, "\n");
      }
    }
  }
  fclose(fpin);
}

void abstract(void)
{
  int l;
  fprintf(fplog,"\n\nABSTRACT:\n");
  fprintf(fplog,"\n |     |          LRN|          VAL|          TST|");
  fprintf(fplog,"\n |    #|   MSE| CLA E|   MSE| CLA E|   MSE| CLA E| SOMs on TST:");
  fprintf(fplog,"\n||%5d, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, ",
    best_epoch,
    best_mse[1], 100.0-best_class[1],
    best_mse[2], 100.0-best_class[2],
    best_mse[3], 100.0-best_class[3]);
  for (l=1; l<=last_layer; l++) {
    if (DIM_SOM[l]>0) { fprintf(fplog, " %d, %5.2f,", l, 100.0-best_som[l][3]); }
  }
  fprintf(fplog, "\n||%5d, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, ",
    epoch,
    end_mse[1], 100.0-end_class[1],
    end_mse[2], 100.0-end_class[2],
    end_mse[3], 100.0-end_class[3]);
  for (l=1; l<=last_layer; l++) {
    if (DIM_SOM[l]>0) { fprintf(fplog, " %d, %5.2f,", l, 100.0-end_som[l][3]); }
  }

  /* open BEST.ENV */
  read_environment(BEST_ENV);
  nothing_done=0;
  reset_bp_confusion_matrix();

  /* null initialisation average activations */
  reset_avg_act();
  reset_std_act();
  count_avg_act(1);

  /* classlabeling epoch */
  epoch=best_epoch;
  clear_som_cell_counters();
  complete_epoch(0, 1);
  count_som_cell_winners();
  count_std_act(example);

  /* dump results and test-epoch */
  dump_som_labeling();
  dump_avg_std();
  DUMP=1;
  reset_bp_confusion_matrix();
  complete_epoch(0, 3);
  dump_bp_confusion_matrix();

  fprintf(fplog, "\n\n |# pruned units: ");
  for (l=1; l<last_layer; l++) {
    fprintf(fplog, "%1d > %1d pruned ", l, best_n_pruned_units[l]);
  }
}

void  reset_bp_confusion_matrix(void)
{
  int i, j;
  for (i=1; i<=n_labels; i++) {
    for (j=1; j<=n_labels; j++) {
      bp_confusion_matrix[i][j]=0.0;
    }
  }
}

void dump_bp_confusion_matrix(void)
{
  int i, j;
  fprintf(fplog, "\n\n |BP-confusion matrix:");
  fprintf(fplog, "\n |        ");
  for (i=1;i<=n_labels;i++) { fprintf(fplog, "%s   ", classlabel[i]); }
  for (i=1;i<=n_labels;i++) {
    fprintf(fplog, "\n | %s  ", classlabel[i]);
    for (j=1;j<=n_labels;j++) {
      fprintf(fplog, " %5d", bp_confusion_matrix[i][j]);
    }
  }
  fprintf(fplog, "\n\n");
}

void make_labels_from_all_labels(void)
{
  int  l, i, pos;
  for (l=1; l<=n_labels; l++) {
    pos=(l-1) * (LABELL+1);
    for (i=0; i<LABELL; i++) {
      classlabel[l][i]=ALL_LABELS[pos+i];
    }
  }
}

void fatal_error(void)
{
  fprintf(fplog, "\n\nEXIT: fatal error");
  fclose(fplog);
  exit(0);
}

void drange_cheque(char *par, int *value,
                   int min, int max, int def)
{
  if ((*value>max) || (*value<min)) {
    printf("\nWarning: %s = %d out of range (%d .. %d);",
           par, *value, min, max);
    fprintf(fplog, "\nWarning: %s = %d out of range (%d .. %d);",
           par, *value, min, max);
    if (def>=0.0) {
      printf("\ndefault %d used.", def);
      fprintf(fplog, "\ndefault %d used.", def);
      *value=def;
    } else {
      printf("\nno default avaiable."); fatal_error();
    }
  }
}

void frange_cheque(char *par, float *value,
                   float min, float max, float def)
{
  if ((*value>max) || (*value<min)) {
    printf("\nWarning: %s = %6.5f out of range (%6.5f .. %6.5f);",
           par, *value, min, max);
    fprintf(fplog, "\nWarning: %s = %6.5f out of range (%6.5f .. %6.5f);",
           par, *value, min, max);
    if (def>=0.0) {
      printf("\ndefault %f used.", def);
      fprintf(fplog, "\ndefault %f used.", def);
      *value=def;
    } else {
      printf("\nno default avaiable."); fatal_error();
    }
  }
}
