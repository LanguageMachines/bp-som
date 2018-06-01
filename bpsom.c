#define VERSION "V0.007"

/*
   BP_SOM.C
   A combination of the traditional back-propagation (BP) learning
   rule with unsupervised learning in self-organizing maps (SOMs) as
   described in Neural Processing Letters (NPL) 2.5 (1995) page xx - xx.
   There are (small) implementation differences between this version
   and the BP_SOM implementation as used for experiments in the NPL-article.

   0.001: prune_weight function (Febr 2 96, Ton Weijters)
   not    start_momentum=0 after pruning
   but    updating del_old
   0.002: use class distribution information (Febr 8 96, Ton Weijters &
          Antal van den Bosch)
   not    class labelling on the basis of the absolute number classes          patterns
   but    relative to the class distribution in the training material
   0.003: prune_weights (Febr 12 96, Antal van den Bosch & Ton Weijters)
   not    prune ONE hidden units with smallest STD < threshold
   but    prune ALL hidden units with STD < threshold
   0.004: record the results on the testing material during the
          abstract epoch (Ton Weijters)
   0.005: Use class distribution information also during online
          class labeling (March 15 1996, Ton Weijters & Antal van den Bosch)
   0.006: Initialisation of act[0] vector to [0 0 ... 0].
          (March 20 1966, Ton Weijters)
   0.007: Calculation of the standard deviation during each learn epoch
          (not during the class labeling epoch). The average_act of the
          epoch before the curent epoch is used during this calculation.
          If ONLINE_CLASS_LABELING = 0 then BATCH_CLASS_LABELING is 
          performed: updating of the labeling on the learning material 
          before the validation epoch. If ONLINE_CLASS_LABELING > 0 
          (for instance 500) the som_cell_tellers are online updated; the
          winners and reliability is calculated after processing 500 
          examples. (Ton Weijters, April 4 1996)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


#define INCLUDE1 "bp.h"
#define INCLUDE2 "som.h"
#define INCLUDE3 "proben1.hlp"

#define PC 1
#define EVER ;;
#define MAX_UNITS 300
#define MAX_LAST_LAYER 5       /* don't change (N_0, ... , N_5)   */
#define NAME_L 128
#define DONOTHING 0.00000001
#define SOM_LR_FALL 1
#define SOM_CONTEXT_FALL 4
#define MIN_IMPROVE 0.005
#define BIAS_CORRECTION 1.00    /* 0.5 is x^2, 0.333 is x^3, etc. */

/* SOM-parameters */

#define M_DIM_SOM_NETWORK 100
#define MAX_LABELS 75
#define LABELL 3             /* don't change (layout in log-file) */
#define CLASS_COUNT_DIV 2

/*********************/
/* BP-SOM functions   */
/*********************/

void  read_parameters_open_log(char *prm_file);
void  allocate_memory(void);
void  count_time(void);
void  complete_pass(int lrn_yn, int material);

void  random_weights(void);
void  open_fpin(char *filename);
void  open_fpout(char *filename);
void  save_environment(char *filename);
void  read_environment(char *filename);
void  fatal_error(void);
void  wait_bpsom(void);
void  abstract(void);
void  log_confusion_matrix(void);

/****************************/
/* INCLUDE1-BP-functions    */
/****************************/

float count_MSE(void);
void  forward(void);
void  backprop(float mm);
float activation_function(float summ);

void  show_result(long int example, long int t_ok);
void  record_result(long int example);

/* monitoring tool 1 */
void reset_average_act(void);
void update_average_act(void);
void ress_average_actc(long int teller);
void reset_ss(void);
void update_ss(void);
void ss_to_std(long int teller);
void record_corr(void);

/* monitoring tool 2 */
void histogram_teller(float weight);
void record_histogram(void);

/* prune tools */
void clear_prune_information(void);
void prune_if_possible(void);
void prune_weights(int l, int w);

/***************************/
/* INCLUDE2 SOM-functions  */
/***************************/

float count_distance(int l, float min_dis, float *v);
void  process_som_vectors(int lrn_yn, int material);
void  update_som_network(int l, int x, int y, int distance);
void  show_som_labeling(void);
void  record_som_labeling(void);

void  clear_som_cell_counters(void);
void  count_som_cell_winners(void);
void  clear_som_conf_matrix(void);
void  show_som_conf_matrix(void);
void  record_som_conf_matrix(void);
char  *take_label(int n);
void  show_soms_epoch_score(void);
void  record_soms_epoch_score(void);
void  div_class_count(void);

/**********************/
/* INCLUDE3-functions */
/**********************/

void  read_input_target(void);
int   res_ok(void);

/***************************/
/* global BP-SOM-variable  */
/***************************/

int   part_win_x[MAX_LAST_LAYER+1], part_win_y[MAX_LAST_LAYER+1],
      use_som_error[MAX_LAST_LAYER+1];
float *weight[MAX_LAST_LAYER][MAX_UNITS+1],
      *del_old[MAX_LAST_LAYER][MAX_UNITS+1],
      *ss_xy[MAX_LAST_LAYER][MAX_UNITS+1];

float *act[MAX_LAST_LAYER+1],
      *target,
      *average_act[MAX_LAST_LAYER+1],
      *average_actc[MAX_LAST_LAYER+1],
      *std_act[MAX_LAST_LAYER+1],
      *ss_act[MAX_LAST_LAYER+1];

int   n_pruned_units[MAX_LAST_LAYER+1];

float *error[MAX_LAST_LAYER+1], best_p_correct=0.0;

int   n[MAX_LAST_LAYER+1], show[MAX_LAST_LAYER+1], column[21],
      epoch=0, start_momentum=0,
      classification_ok=0, nothing_done=0;
long int example=0,
         debug_teller=0,
         debug_factor=0,
         som_error_use_counter[MAX_LAST_LAYER+1];

int   winner_reliability[MAX_LAST_LAYER+1];
float winner_dis[MAX_LAST_LAYER+1];
int   part_winner_reliability[MAX_LAST_LAYER+1];
float part_winner_dis[MAX_LAST_LAYER+1];
int   winner_labeln[MAX_LAST_LAYER+1];

      /* abstract variables */
int   best_epoch=1;
float best_class[4], end_class[4],
      best_mse[4], end_mse[4],
      best_som[MAX_LAST_LAYER+1][4], end_som[MAX_LAST_LAYER+1][4];
int   best_n_pruned_units[MAX_LAST_LAYER+1];

int   confusion_matrix[MAX_LABELS+1][MAX_LABELS+1];

FILE *fpin, *fpout, *fplog;

/* BP-SOM-parameters */

float BP_SOM_RATIO;

/* BP parameters */
int   SEED, N_0, N_1, N_2, N_3, N_4, N_5, LAST_LAYER,
      DEBUG, RECORD, HISTOGRAM,
      SHOW_0, SHOW_1, SHOW_2, SHOW_3, SHOW_4, SHOW_5,
      N_EPOCHS, TEST_RATIO, STOP_IF_NOTHING_DONE, CORR,
      MM_AFTER_N_EXAMPLES;
long int ONLINE_CLASS_LABELING;
float LEARN_RATE, MOMENTUM, GOOD_FALSE_TOLERANCE,
      FLAT_SPOT_ELIMINATION, UPDATE_TOLERANCE, PRUNE_THRESHOLD;
char  parameters[NAME_L], LRN[NAME_L], VAL[NAME_L], TST[NAME_L],
      LOG[NAME_L], OLD_ENV[NAME_L], BEST_ENV[NAME_L];

/****************************/
/* global SOM-variables */
/****************************/

float    *som_network_vector[MAX_LAST_LAYER+1]
           [M_DIM_SOM_NETWORK+1][M_DIM_SOM_NETWORK+1],
         *som_vector[MAX_LAST_LAYER+1],
         som_total_distance[MAX_LAST_LAYER+1], som_lr;
int      som_cont, n_labels;
long int som_ok[MAX_LAST_LAYER+1];
long int *som_cell_teller[MAX_LAST_LAYER+1]
                       [M_DIM_SOM_NETWORK+1]
                       [M_DIM_SOM_NETWORK+1];

long int *som_conf_matrix[MAX_LAST_LAYER+1][MAX_LABELS+1];
int      *som_winner_percents[MAX_LAST_LAYER+1][M_DIM_SOM_NETWORK+1];
int      DIM_SOM_0, DIM_SOM_1, DIM_SOM_2,
         DIM_SOM_3, DIM_SOM_4, DIM_SOM_5;
int      labeln, helplabeln;
char     labels[LABELL+1], helplabels[LABELL+1];
long int bias_counter[MAX_LABELS+1]; /* bias_counter[0]=total */
float    bias[MAX_LABELS+1];

/* SOM-parameters */
int   DIM_SOM[MAX_LAST_LAYER+1], CONTEXT, MIN_CONTEXT,
      USE_SOM_ERROR_THRESHOLD;
float SOM_LEARN_RATE, MIN_LEARN_RATE;
char  ALL_LABELS[256];


#include INCLUDE1
#include INCLUDE2
#include INCLUDE3

int main(argc, argv)
int argc;
char **argv;
{
  int i;
  char buff[NAME_L];
  strcpy(labels, "---");
  setbuf(stdout, NULL);
  if (argc>=2) {
    read_parameters_open_log(argv[1]);
  } else {
    printf("\nName parameter file: ");
    gets(buff);
    read_parameters_open_log(buff);
  }
  if (strlen(OLD_ENV)>1) {
    read_environment(OLD_ENV);
  } else {
    allocate_memory();
    if (argc==3) {
      SEED=atoi(argv[2]);
      fprintf(fplog, "\nSEED=%d", SEED);
    }
    srand(SEED);
    random_weights();
    clear_som_cell_counters();
    for (i=1; i<=n[0]; i++) { act[0][i]=0.0; };
  }
  if (PRUNE_THRESHOLD>DONOTHING) { clear_prune_information(); }
  for (i=0;i<=MAX_LABELS;i++) { bias_counter[i]=bias[i]=0.0; }

  for (EVER) {
    epoch++; nothing_done++;
    if ((epoch > N_EPOCHS) || (nothing_done>=STOP_IF_NOTHING_DONE)) {
      abstract();
      count_time();
      fprintf(fplog, "\n\nBP_SOM-N ready");
      fclose(fplog);
      exit(0);
    }

    /* update SOM-context */
    som_cont=floor( (pow((double) N_EPOCHS-epoch,
		                   (double) SOM_CONTEXT_FALL)/
                       pow((double) N_EPOCHS,
		                   (double) SOM_CONTEXT_FALL)) *
               (CONTEXT-MIN_CONTEXT) + 0.5) +  MIN_CONTEXT;

    /* update SOM-learning-rate */
    som_lr=(float) ((pow((double) N_EPOCHS-epoch,
		                     (double) SOM_LR_FALL)/
                      pow((double) N_EPOCHS,
		                  (double) SOM_LR_FALL)) *
                      (SOM_LEARN_RATE-MIN_LEARN_RATE)) +
               MIN_LEARN_RATE;
    if (CORR || (PRUNE_THRESHOLD>DONOTHING)) {
      reset_average_act();
      reset_ss();
    }
    if (ONLINE_CLASS_LABELING>0) {
      div_class_count();
    } 
    if (LEARN_RATE>DONOTHING) {  complete_pass(1, 1); }
    if ((epoch % TEST_RATIO)==0) {
      if (CORR || (PRUNE_THRESHOLD>DONOTHING)) {
	ress_average_actc(example);
        ss_to_std(example);
      }
      if (ONLINE_CLASS_LABELING==0) {
        if (CORR || (PRUNE_THRESHOLD>DONOTHING)) {
	  reset_average_act();
          reset_ss();
        }
        clear_som_cell_counters();
        complete_pass(0, 1);
	count_som_cell_winners();
        if (CORR || (PRUNE_THRESHOLD>DONOTHING)) {
	  ress_average_actc(example);
          ss_to_std(example);
        }
      } 
      record_som_labeling();
      if (DEBUG) { show_som_labeling(); }
      if (strlen(VAL)>1) { complete_pass(0, 2); }
      if (CORR || (PRUNE_THRESHOLD>DONOTHING)) { record_corr(); }
      if (PRUNE_THRESHOLD>DONOTHING) {
        prune_if_possible();
      }
      if (strlen(TST)>1) { complete_pass(0, 3); }
    }
  }
}

void complete_pass(int lrn_yn, int material)
{
  long int tel_ok=0;
  int i,l;
  float pass_error=0.0, p_correct=0.0;
  example=0;
  for (l=0;l<=LAST_LAYER;l++) {
    som_total_distance[l]=0.0; som_ok[l]=0; som_error_use_counter[l]=0;
  }
  if (material==1) { open_fpin(LRN); }
  if (material==2) { open_fpin(VAL); }
  if (material==3) { open_fpin(TST); }
  if (PC) { printf("\nEpoch %d/%d Type %d %d",
              epoch, N_EPOCHS, lrn_yn, material); }
  clear_som_conf_matrix();

  for (EVER) {
    read_input_target(); example++;
    if ((epoch==1) && (lrn_yn==1)) {
      bias_counter[0]++; bias_counter[labeln]++;
    }
    forward();
    if (res_ok()==1) {
      classification_ok=1; tel_ok++;
    } else {
      classification_ok=0;
    }
    pass_error+=count_MSE();

    if ((lrn_yn==1) &&
	(ONLINE_CLASS_LABELING>0) &&
	(epoch>TEST_RATIO) &&
	((example % ONLINE_CLASS_LABELING)==0)) {
      count_som_cell_winners();
    }
    process_som_vectors(lrn_yn, material);
    if (DEBUG) show_result(example, tel_ok);
    if (RECORD) record_result(example);
    if (material==1) {
      if (((epoch % TEST_RATIO)==0) &&
	  (CORR || (PRUNE_THRESHOLD>DONOTHING))) {
            update_average_act();
            update_ss();
      }
    }
    if (lrn_yn==1) {
      if (start_momentum>=MM_AFTER_N_EXAMPLES) {
        backprop(MOMENTUM);
      } else {
        backprop(0.0);
        start_momentum++;
      }
    }
    if (DEBUG) {
          printf("\n============================================");
          wait_bpsom();
    }
    if (feof(fpin)) {
      fclose(fpin);
      if (example>0) {
        som_total_distance[l]=(long int) (som_total_distance[l]/example);
	if ((epoch==1) && (lrn_yn==1)) {
	  fprintf(fplog, "\n\nClass distribution in learning material:\n");
	  for (i=1; i<=n_labels; i++) { fprintf(fplog, "%5d", i); }
	  fprintf(fplog, "\n");
	  for (i=1; i<=n_labels; i++) {
	    bias[i]=(float) (1.0*bias_counter[i]/bias_counter[0]);
	    fprintf(fplog, "%5.2f", bias[i]);
          }
	  fprintf(fplog, "\n");
        }
      }

      fprintf(fplog, "\nLRN=%d-MAT=%d ", lrn_yn, material);

      pass_error=(pass_error/(example*n[LAST_LAYER]));
      p_correct=(100.0*tel_ok)/example;
      if (PC) {
        if (DEBUG) { printf("\n"); }
	printf(" Class_E: %6.2f, SOM-error use:", 100.0-p_correct);
        for (l=1;l<=LAST_LAYER;l++) {
	  if (DIM_SOM[l]>0) {
	    printf(" %5.1f,", (100.0*som_error_use_counter[l]/example));
	  }
        }
      }
      end_class[material]=p_correct;
      end_mse[material]=pass_error;
      for (l=0; l<=LAST_LAYER; l++) {
        end_som[l][material]=100.0*som_ok[l]/example;
      }
      fprintf(fplog,
	"epoch: %5d, MSErr: %5.3f, Class_E: %6.2f, SOM-error use: ",
        epoch, pass_error, 100.0-p_correct);
      for (l=1;l<=LAST_LAYER;l++) {
	if (DIM_SOM[l]>0) {
	  fprintf(fplog,
	  " %5.1f,", (100.0*som_error_use_counter[l]/example));
	}
      }
      if (material==2) {
        if ((p_correct>best_p_correct) ||
            ((p_correct==best_p_correct) &&
             (pass_error+MIN_IMPROVE<best_mse[2]))) {
          best_p_correct=p_correct;
          best_epoch=epoch;
          if (strlen(BEST_ENV)>1) {
            if (PC) { printf("*"); }
            fprintf(fplog, " * saved!");
            save_environment(BEST_ENV);
            nothing_done=0;
          }
        }
        if (HISTOGRAM) { record_histogram(); }
      }
      save_environment("last.env");
      if ((material==3) && (epoch==best_epoch)) {
        for (i=1; i<=3; i++) {
          best_class[i]=end_class[i];
          best_mse[i]=end_mse[i];
          for (l=0; l<=LAST_LAYER; l++) {
            best_som[l][i]=end_som[l][i];
          }
          for (l=1; l<LAST_LAYER; l++) {
            best_n_pruned_units[l]=n_pruned_units[l];
          }
        }
      }
      if ((RECORD) || (material==3)) {record_soms_epoch_score(); }
      if (DEBUG) {show_soms_epoch_score(); }
      if ((DEBUG) || (material>1)) { record_som_conf_matrix(); }
      break;
    }
  }
}

void read_parameters_open_log(char *prm_file)
{
  int i;
  if ((fpin=fopen(prm_file, "r"))==NULL) {
    printf("\nOpening parameterfile %s failed", prm_file);
    exit(0);
  }
  /*** read bp_n-parameters ***/
  fscanf(fpin, "SEED = %d", &SEED);
  fscanf(fpin, "\nN_0 = %d", &N_0);
  fscanf(fpin, "\nN_1 = %d", &N_1);
  fscanf(fpin, "\nN_2 = %d", &N_2);
  fscanf(fpin, "\nN_3 = %d", &N_3);
  fscanf(fpin, "\nN_4 = %d", &N_4);
  fscanf(fpin, "\nN_5 = %d", &N_5);

  fscanf(fpin, "\nDEBUG = %d", &DEBUG);
  fscanf(fpin, "\nRECORD = %d", &RECORD);
  fscanf(fpin, "\n  SHOW_0 = %d", &SHOW_0);
  fscanf(fpin, "\n  SHOW_1 = %d", &SHOW_1);
  fscanf(fpin, "\n  SHOW_2 = %d", &SHOW_2);
  fscanf(fpin, "\n  SHOW_3 = %d", &SHOW_3);
  fscanf(fpin, "\n  SHOW_4 = %d", &SHOW_4);
  fscanf(fpin, "\n  SHOW_5 = %d", &SHOW_5);

  fscanf(fpin, "\nPRUNE_THRESHOLD =%f", &PRUNE_THRESHOLD);
  fscanf(fpin, "\nCORR =%d", &CORR);
  fscanf(fpin, "\nHISTOGRAM =%d", &HISTOGRAM);

  fscanf(fpin, "\nN_EPOCHS = %d", &N_EPOCHS);
  fscanf(fpin, "\nSTOP_IF_NOTHING_DONE = %d", &STOP_IF_NOTHING_DONE);
  fscanf(fpin, "\nTEST_RATIO = %d", &TEST_RATIO);
  fscanf(fpin, "\nLEARN_RATE = %f", &LEARN_RATE);
  fscanf(fpin, "\nMOMENTUM = %f", &MOMENTUM);
  fscanf(fpin, "\nMM_AFTER_N_EXAMPLES = %d", &MM_AFTER_N_EXAMPLES);

  fscanf(fpin, "\nGOOD_FALSE_TOLERANCE = %f", &GOOD_FALSE_TOLERANCE);
  fscanf(fpin, "\nFLAT_SPOT_ELIMINATION = %f", &FLAT_SPOT_ELIMINATION);
  fscanf(fpin, "\nUPDATE_TOLERANCE = %f", &UPDATE_TOLERANCE);

  fscanf(fpin, "\nLRN = %s", LRN);
  fscanf(fpin, "\nVAL = %s", VAL);
  fscanf(fpin, "\nTST = %s", TST);
  fscanf(fpin, "\nLOG = %s", LOG);
  fscanf(fpin, "\nOLD_ENV = %s", OLD_ENV);
  fscanf(fpin, "\nBEST_ENV = %s", BEST_ENV);

  /*** read the bp_som parameter ***/
  fscanf(fpin, "\nBP_SOM_RATIO = %f", &BP_SOM_RATIO);
  fscanf(fpin, "\nUSE_SOM_ERROR_THRESHOLD = %d", &USE_SOM_ERROR_THRESHOLD);

  /*** read som_n parameters ***/
  fscanf(fpin, "\nDIM_SOM_0 = %d", &DIM_SOM_0);
  fscanf(fpin, "\nDIM_SOM_1 = %d", &DIM_SOM_1);
  fscanf(fpin, "\nDIM_SOM_2 = %d", &DIM_SOM_2);
  fscanf(fpin, "\nDIM_SOM_3 = %d", &DIM_SOM_3);
  fscanf(fpin, "\nDIM_SOM_4 = %d", &DIM_SOM_4);
  fscanf(fpin, "\nDIM_SOM_5 = %d", &DIM_SOM_5);
  fscanf(fpin, "\nONLINE_CLASS_LABELING = %ld", &ONLINE_CLASS_LABELING);
  fscanf(fpin, "\nCONTEXT = %d", &CONTEXT);
  fscanf(fpin, "\nMIN_CONTEXT = %d", &MIN_CONTEXT);
  fscanf(fpin, "\nALL_LABELS = %s", ALL_LABELS);
  n_labels = (int) ((strlen(ALL_LABELS)+1)/(LABELL+1));
  if ((n_labels>MAX_LABELS) || (strlen(ALL_LABELS)+1>256)) {
    printf("\nFatal error: to many labels %d (max = %d)",
            n_labels, MAX_LABELS);
    exit(0);
  }
  fscanf(fpin, "\nSOM_LEARN_RATE = %f", &SOM_LEARN_RATE);
  fscanf(fpin, "\nMIN_LEARN_RATE = %f", &MIN_LEARN_RATE);

  fclose(fpin);
  if (strlen(LOG)<=1) {
    strncpy(LOG, prm_file, strlen(prm_file)-4);
    strcat(LOG, ".log");
  }
  if ((fplog = fopen(LOG, "a")) == NULL) {
    printf("\nFatal error: open log file %s failed!", LOG);
    exit(0);
  }
  setbuf(fplog, NULL);

  fprintf(fplog, "\nNEW SESSION BP-SOM %s (Copyright 1995 ton weijters)",
                 VERSION); count_time();

  fprintf(fplog, "\nSEED = %d\n", SEED);

  n[0]=N_0; n[1]=N_1; n[2]=N_2; n[3]=N_3; n[4]=N_4; n[5]=N_5;
  show[0]=SHOW_0; show[1]=SHOW_1; show[2]=SHOW_2;
  show[3]=SHOW_3; show[4]=SHOW_4; show[5]=SHOW_5;

  for (i=0; i<=MAX_LAST_LAYER; i++) {
    fprintf(fplog, "\nN[%d] = %3d       (max=%d)", i, n[i], MAX_UNITS);
  }
  fprintf(fplog, "\n\nDEBUG = %d        (0=off, 1=on)", DEBUG);
  fprintf(fplog, "\nRECORD = %d       (0=off, 1=on)", RECORD);
  fprintf(fplog, "\n  SHOW_0 = %d       (0=off, 1=on)", SHOW_0);
  fprintf(fplog, "\n  SHOW_1 = %d       (0=off, 1=on)", SHOW_1);
  fprintf(fplog, "\n  SHOW_2 = %d       (0=off, 1=on)", SHOW_2);
  fprintf(fplog, "\n  SHOW_3 = %d       (0=off, 1=on)", SHOW_3);
  fprintf(fplog, "\n  SHOW_4 = %d       (0=off, 1=on)", SHOW_4);
  fprintf(fplog, "\n  SHOW_5 = %d       (0=off, 1=on)", SHOW_5);
  fprintf(fplog, "\nPRUNE_THRESHOLD = %f (0.0=off, >0.0=Prune Threshold)",
    PRUNE_THRESHOLD);

  fprintf(fplog, "\nCORR = %d         (0=off, 1=on)", CORR);
  fprintf(fplog, "\nHISTOGRAM = %d    (0=off, 1=on)", HISTOGRAM);

  fprintf(fplog, "\n\nN_EPOCHS = %d", N_EPOCHS);
  fprintf(fplog, "\nSTOP_IF_NOTHING_DONE = %d", STOP_IF_NOTHING_DONE);
  fprintf(fplog, "\nTEST_RATIO = %d", TEST_RATIO);

  fprintf(fplog, "\n\nLEARN_RATE = %f", LEARN_RATE);
  fprintf(fplog, "\nMOMENTUM = %f", MOMENTUM);
  fprintf(fplog, "\nMM_AFTER_N_EXAMPLES = %d (use momentum after # examples)",
    MM_AFTER_N_EXAMPLES);

  fprintf(fplog, "\n\nGOOD_FALSE_TOLERANCE = %f", GOOD_FALSE_TOLERANCE);
  fprintf(fplog, "\nFLAT_SPOT_ELIMINATION = %f", FLAT_SPOT_ELIMINATION);
  fprintf(fplog, "\nUPDATE_TOLERANCE = %f", UPDATE_TOLERANCE);

  fprintf(fplog, "\n\nLRN = %s", LRN);
  fprintf(fplog, "\nVAL = %s", VAL);
  fprintf(fplog, "\nTST = %s", TST);
  fprintf(fplog, "\nLOG = %s", LOG);
  fprintf(fplog, "\nOLD_ENV = %s", OLD_ENV);
  fprintf(fplog, "\nBEST_ENV = %s", BEST_ENV);

  for (i=0; i<=MAX_LAST_LAYER; i++) {
    if (n[i]>0) {
      LAST_LAYER=i;
    }
    if (n[i]>MAX_UNITS) {
      fprintf(fplog,
	      "\nN_%d = %d: Fatal error; impossible BP architecture!",
              i, n[i]);
      fatal_error();
    }
  }
  fprintf(fplog, "\n\nBP_SOM_RATIO = %f", BP_SOM_RATIO);
  fprintf(fplog, "\nUSE_SOM_ERROR_THRESHOLD = %d", USE_SOM_ERROR_THRESHOLD);

  DIM_SOM[0]=DIM_SOM_0; DIM_SOM[1]=DIM_SOM_1;
  DIM_SOM[2]=DIM_SOM_2; DIM_SOM[3]=DIM_SOM_3;
  DIM_SOM[4]=DIM_SOM_4; DIM_SOM[5]=DIM_SOM_5;
  for (i=0; i<=MAX_LAST_LAYER; i++) {
    if (DIM_SOM[i]>M_DIM_SOM_NETWORK) {
      fprintf(fplog,
	      "\nN_%d = %d: Fatal error; impossible SOM architecture!",
              i, DIM_SOM[i]);
      fatal_error();
    }
  }
  fprintf(fplog, "\nDIM_SOM_0 = %d", DIM_SOM[0]);
  fprintf(fplog, "\nDIM_SOM_1 = %d", DIM_SOM[1]);
  fprintf(fplog, "\nDIM_SOM_2 = %d", DIM_SOM[2]);
  fprintf(fplog, "\nDIM_SOM_3 = %d", DIM_SOM[3]);
  fprintf(fplog, "\nDIM_SOM_4 = %d", DIM_SOM[4]);
  fprintf(fplog, "\nDIM_SOM_5 = %d", DIM_SOM[5]);

  fprintf(fplog, "\nONLINE_CLASS_LABELING = %ld", ONLINE_CLASS_LABELING);
  fprintf(fplog, "\nCONTEXT = %d", CONTEXT);
  fprintf(fplog, "\nMIN_CONTEXT = %d", MIN_CONTEXT);
  fprintf(fplog, "\nALL_LABELS = %s", ALL_LABELS);

  fprintf(fplog, "\nSOM_LEARN_RATE = %f", SOM_LEARN_RATE);
  fprintf(fplog, "\nMIN_LEARN_RATE = %f", MIN_LEARN_RATE);
}

void allocate_memory(void)
{
  int l, i, j;

  for (l=0;l<LAST_LAYER;l++) {
    for (i=0;i<=n[l];i++) {
      weight[l][i]=(float *) calloc((unsigned) n[l+1]+1, sizeof(float));
      del_old[l][i]=(float *) calloc((unsigned) n[l+1]+1, sizeof(float));
      if ((l>0) && (l<LAST_LAYER) && (CORR || (PRUNE_THRESHOLD>DONOTHING))){
         ss_xy[l][i]=(float *) calloc((unsigned) n[l]+1, sizeof(float));
      }
    }
  }

  for (l=0;l<=LAST_LAYER;l++) {
    act[l]=(float *) calloc((unsigned) n[l]+1, sizeof(float));
    if (l>0) {
      error[l]=(float *) calloc((unsigned) n[l]+1, sizeof(float));
    }
    if ((l>0) && (l<LAST_LAYER) && (CORR || (PRUNE_THRESHOLD>DONOTHING))){
      average_act[l]=(float *) calloc((unsigned) n[l]+1, sizeof(float));
      average_actc[l]=(float *) calloc((unsigned) n[l]+1, sizeof(float));
      std_act[l]=(float *) calloc((unsigned) n[l]+1, sizeof(float));
      ss_act[l]=(float *) calloc((unsigned) n[l]+1, sizeof(float));
    }
  }

  for (l=0;l<=LAST_LAYER;l++) {
    for (i=1;i<=DIM_SOM[l];i++) {
      for (j=1;j<=DIM_SOM[l];j++) {
        som_network_vector[l][i][j] =
              (float *) calloc((unsigned) n[l]+1, sizeof(float));
        som_cell_teller[l][i][j] =
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
    som_vector[l] =
        (float *) calloc((unsigned) n[l]+1, sizeof(float));
  }
  target=(float *) calloc((unsigned) n[LAST_LAYER]+1, sizeof(float));
  if (target==NULL){
    printf("\nOut of memory. Allocation failed!");
    exit(0);
  }
}

void count_time(void)
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
    fprintf(fplog, "\nOpen fpin as %s failed!", filename);
  }
}

void open_fpout(char *filename)
{
  if ((fpout=fopen(filename, "w"))==NULL) {
    fprintf(fplog, "\nCannot open %s.", filename);
  }
}

void random_weights(void)
{
  int l,i,j,k;
  float random_average=0.0;

  for (i=1; i<=1000; i++) {
    random_average+=rand()/1000;
  }

  for (l=0; l<LAST_LAYER; l++) {
    for (i=0; i<=n[l]; i++) {
      for (j=1; j<=n[l+1]; j++) {
        weight[l][i][j]=(float) (random_average - rand())/
                                  random_average;
      }
    }
  }

  for (l=0; l<LAST_LAYER; l++) {
    for (i=1;i<=DIM_SOM[l];i++) {
      for (j=1;j<=DIM_SOM[l];j++) {
        for (k=1; k<=n[l]; k++) {
          som_network_vector[l][i][j][k] =
            (0.5 + (0.1 * (float) (random_average - rand())/
                                   random_average));
        }
      }
    }
  }
}

void save_environment(char *filename)
{
  int i,j,k,l;
  open_fpout(filename);
  fprintf(fpout, "%d %d %d %d %d %d\n",
                n[0], n[1], n[2], n[3], n[4], n[5]);
  fprintf(fpout, "%d %d %d %d %d %d\n",
                DIM_SOM_0, DIM_SOM_1, DIM_SOM_2,
                DIM_SOM_3, DIM_SOM_4, DIM_SOM_5);

  for (i=0; i<LAST_LAYER; i++) {
    for (j=0; j<=n[i]; j++) {
      for (k=1; k<=n[i+1]; k++) {
        fprintf(fpout, "%7.4f ", weight[i][j][k]);
      }
      fprintf(fpout, "\n");
    }
  }

  /* write SOM-weights */
  for (l=0; l<=LAST_LAYER; l++) {
    for (i=1; i<=DIM_SOM[l]; i++) {
      for (j=1; j<=DIM_SOM[l]; j++) {
        for (k=1; k<=n[l]; k++) {
          fprintf(fpout, "%f ", som_network_vector[l][i][j][k]);
        }
        fprintf(fpout, "\n");
      }
    }
  }
  fclose(fpout);
}

void read_environment(char *filename)
{
  int l,i,j,k;

  open_fpin(filename);
  fscanf(fpin, "%d %d %d %d %d %d\n", &N_0, &N_1, &N_2, &N_3, &N_4, &N_5);
  n[0]=N_0; n[1]=N_1; n[2]=N_2; n[3]=N_3; n[4]=N_4; n[5]=N_5;
  fprintf(fplog, "\n\nNew network format:");
  fprintf(fplog, "\n%d %d %d %d %d %d",
                 n[0], n[1], n[2], n[3], n[4], n[5]);

  fscanf(fpin, "%d %d %d %d %d %d\n", &DIM_SOM_0, &DIM_SOM_1, &DIM_SOM_2,
                &DIM_SOM_3, &DIM_SOM_4, &DIM_SOM_5);
  DIM_SOM[0]=DIM_SOM_0; DIM_SOM[1]=DIM_SOM_1; DIM_SOM[2]=DIM_SOM_2;
  DIM_SOM[3]=DIM_SOM_3; DIM_SOM[4]=DIM_SOM_4; DIM_SOM[5]=DIM_SOM_5;
  fprintf(fplog, "\nDIM_SOMS = %d %d %d %d %d %d",
                 DIM_SOM_0, DIM_SOM_1, DIM_SOM_2,
                DIM_SOM_3, DIM_SOM_4, DIM_SOM_5);

  for (l=0; l<=MAX_LAST_LAYER; l++) {
    if (n[l]>0) { LAST_LAYER=l; }
    if ((n[l]>MAX_UNITS) || (DIM_SOM[l]>M_DIM_SOM_NETWORK)) {
      fprintf(fplog,
	      "\nN_%d = %d %d: Fatal error; impossible network architecture!",
              l, n[l], DIM_SOM[l]);
      fatal_error();
    }
  }
  fprintf(fplog, "\nLAST_LAYER = %d", LAST_LAYER);

  if (epoch==0) { allocate_memory(); }

  /* read SOM-weights */
  for (i=0; i<LAST_LAYER; i++) {
    for (j=0; j<=n[i]; j++) {
      for (k=1; k<=n[i+1]; k++) {
        fscanf(fpin, "%f ", &weight[i][j][k]);
      }
    }
  }

  for (l=0; l<=LAST_LAYER; l++) {
    for (i=1; i<=DIM_SOM[l]; i++) {
      for (j=1; j<=DIM_SOM[l]; j++) {
        for (k=1; k<=n[l]; k++) {
          fscanf(fpin, "%f ", &som_network_vector[l][i][j][k]);
          if (DEBUG) { printf("%4.2f ", som_network_vector[l][i][j][k]); }
        }
        if (DEBUG) { printf("\n"); }
        fscanf(fpin, "\n");
      }
    }
  }
  fclose(fpin);
}


void wait_bpsom(void)
{
  char buff[20];
  debug_teller++;
  if (debug_teller>debug_factor) {
    printf("\n# of steps (-1 means stop debugging): ");
    debug_factor=atoi(gets(buff));
    debug_teller=1;
     if (debug_factor<0) { DEBUG=0; }
  }
}

void abstract(void)
{
  int i,j,l;
  fprintf(fplog,
    "\n |     |          LRN|          VAL|          TST|");
  fprintf(fplog,
    "\n |    #|   MSE| CLA E|   MSE| CLA E|   MSE| CLA E| SOMs on TST:");
  fprintf(fplog,
    "\n||%5d, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, ",
    best_epoch,
    best_mse[1], 100.0-best_class[1],
    best_mse[2], 100.0-best_class[2],
    best_mse[3], 100.0-best_class[3]);
  for (l=1; l<=LAST_LAYER; l++) {
    if (DIM_SOM[l]>0) {
      fprintf(fplog, " %d, %5.2f,", l, 100.0-best_som[l][3]);
    }
  }
  fprintf(fplog,
    "\n||%5d, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, ",
    epoch,
    end_mse[1], 100.0-end_class[1],
    end_mse[2], 100.0-end_class[2],
    end_mse[3], 100.0-end_class[3]);
  for (l=1; l<=LAST_LAYER; l++) {
    if (DIM_SOM[l]>0) {
      fprintf(fplog, " %d, %5.2f,", l, 100.0-end_som[l][3]);
    }
  }
  /* open BEST.ENV */
  read_environment(BEST_ENV);
  nothing_done=0;
  for (i=1; i<=n_labels; i++) {
    for (j=1; j<=n_labels; j++) {
      confusion_matrix[i][j]=0;
    }
  }
  /* count classlabels and reliability */
  complete_pass(0, 1);
  /* record results */
  RECORD=1;
  complete_pass(0, 3);
  log_confusion_matrix();

  fprintf(fplog, "\n|# pruned units: ");
  for (l=1; l<LAST_LAYER; l++) {
    fprintf(fplog, "%1d > %1d pruned ", l, best_n_pruned_units[l]);
  }
}

void log_confusion_matrix(void)
{
  int i,j;
  fprintf(fplog, "\n\n\n|CONFUSION MATRIX:\n\n| ");
  fprintf(fplog, "        ");
  for (i=1;i<=n_labels;i++) { fprintf(fplog, "%s   ", take_label(i)); }
  for (i=1;i<=n_labels;i++) {
    fprintf(fplog, "\n| %s  ", take_label(i));
    for (j=1;j<=n_labels;j++) {
      fprintf(fplog, " %5d", confusion_matrix[i][j]);
    }
  }
  fprintf(fplog, "\n\n");
}

