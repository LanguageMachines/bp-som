/* BP_P.H */

float count_MSE(void)
{
  int i;
  float error=0.0;

  for (i=1; i < (n[LAST_LAYER]+1); i++) {
    error+=((target[i]-act[LAST_LAYER][i]) *
                  (target[i]-act[LAST_LAYER][i]));
  }
  return (float) sqrt((double) error);
}

float activation_function(float summ)
{
  if (summ<-15.0) {
    return(0.0);
  } else {
    if (summ>15.0) {
      return(1.0);
    } else {
      return(1.0 / (1.0 + (float) exp(- (double) summ )));
    }
  }
}

void forward(void)
{
  register int l, i, j;
  register float summ;

  for (l=0; l<LAST_LAYER; l++) {
    act[l][0]=1.0;
    for (j=1; j<=n[l+1]; j++) {
      summ=0.0;
      for (i=0; i<=n[l]; i++) {
	summ+=(weight[l][i][j] * act[l][i]);
      }
      act[l+1][j]=activation_function(summ);
    }
  }
}

void backprop(float mm)
{
  int i, j, k;
  float bp_error, som_error;

  /* output errors */

  for (j=1; j<n[LAST_LAYER]+1; j++) {
    if (fabs(target[j] - act[LAST_LAYER][j])>=UPDATE_TOLERANCE) {
      error[LAST_LAYER][j] =
        (FLAT_SPOT_ELIMINATION +
          (act[LAST_LAYER][j] *  (1.0 - act[LAST_LAYER][j]))) *
          (target[j] - act[LAST_LAYER][j]);
    } else {
      error[LAST_LAYER][j]=0.0;
    }
  }

  /* STEP 6 hidden errors */
  for (i=LAST_LAYER-1; i>0; i--) {
    if (DEBUG) {
      printf("\nBP_SOM_UPDATE with som position [%d, %d]:",
            part_win_x[i], part_win_y[i]);
      printf("\nLayer    | Dimension | BP_error | SOM_error | TOTAL_error");
    }
    if ((BP_SOM_RATIO < 0.999) &&
	(use_som_error[i]==1)  &&
	(epoch>TEST_RATIO) &&
	(part_winner_reliability[i]>=USE_SOM_ERROR_THRESHOLD)) {
      som_error_use_counter[i]++;
    }
    for (j=1; j<=n[i]; j++) {
      if (fabs(act[i][j])>DONOTHING) {
	bp_error=0.0;
        for (k=1; k<=n[i+1]; k++) {
	  bp_error+=(error[i+1][k] * weight[i][j][k]);
	}
	bp_error=bp_error * act[i][j] * (1.0 - act[i][j]);
	/* CALCULATION  OF THE SOM-ERROR */
        if ((BP_SOM_RATIO < 0.999) &&
            (use_som_error[i]==1)  &&
	    (epoch>TEST_RATIO) &&
	    (part_winner_reliability[i]>=USE_SOM_ERROR_THRESHOLD)) {
          som_error=0.01 * part_winner_reliability[i] *
            (som_network_vector[i][part_win_x[i]][part_win_y[i]][j] - act[i][j]);
        } else {
          som_error=0.0;
        }
        if (fabs(som_error)>0) {
	  error[i][j]=(BP_SOM_RATIO * bp_error) +
			((1.0 - BP_SOM_RATIO) * som_error);
        } else {
	  error[i][j]=bp_error;
        }
      } else {
        error[i][j]=som_error=bp_error=0.0;
      }
      if (DEBUG) {
        printf("\n%9d %9d %9.4f %9.4f %9.4f",
        i, j, bp_error, som_error, error[i][j]) ;
      }
    }
  }

  /* STEP 7 en STEP 8 updating the connection weights */
  for (i=0; i<=LAST_LAYER-1; i++) {
    for (j=1; (j<n[i+1]+1); j++) {
      if (fabs(error[i+1][j])>DONOTHING) {
        for (k=0; (k<n[i]+1); k++) {
          if (fabs(act[i][k])>DONOTHING) {
            del_old[i][k][j]=(LEARN_RATE * error[i+1][j] * act[i][k]) +
                               (mm * del_old[i][k][j]);
            weight[i][k][j]+=del_old[i][k][j];
          }
        }
      }
    }
  }
}

void show_result(long int example, long int tel_ok)
{
  int l, i;
  printf("\n\nepoch=%d, #example=%ld, BPok=%d, #ok=%ld",
         epoch, example, classification_ok, tel_ok);
  for (l=0; l<LAST_LAYER; l++) {
    if (show[l]>0) {
      printf("\nL%d: ",l);
      if ((l==0) && (((int) act[l][1])==act[l][1])) {
        for (i=1; i < n[l]+1; i++) {
          printf(" %1.0f", act[l][i]);
        }
      } else {
        for (i=1; i < n[l]+1; i++) {
          printf(" %6.4f", act[l][i]);
        }
      }
      if (DIM_SOM[l]>0) {
        printf(" SOM: %3s, %3d p, Dis %5.2f, Part_winn[%2d,%2d]",
          take_label(winner_labeln[l]), part_winner_reliability[l],
          winner_dis[l], part_win_x[l], part_win_y[l]);
      }
    }
  }
  if (show[LAST_LAYER]>0) {
    printf("\nL%d: ",l);
    for (i=1; i < n[l]+1; i++) {
      printf(" %6.4f", act[l][i]);
    }
    if (DIM_SOM[LAST_LAYER]>0) {
      printf(" SOM: %3s, %3d p, Dist %5.2f",
        take_label(winner_labeln[LAST_LAYER]),
        part_winner_reliability[LAST_LAYER], winner_dis[LAST_LAYER]);
    }
    printf("\nTT: ");
    for (i=1; i < n[LAST_LAYER]+1; i++) {
      printf(" %6.4f", target[i]);
    }
    printf(" %3s", take_label(labeln));
  }
}

void record_result(long int example)
{
  int l, i;
  if (show[1]>0) {
    fprintf(fplog, "\n#<%d,%ld,%d>", epoch, example, classification_ok);
  }
  for (l=0; l<=LAST_LAYER; l++) {
    if (show[l]>0) {
      fprintf(fplog, "\n{%d",l);
      if (DIM_SOM[l]>0) {
        fprintf(fplog, "[%2d,%2d]", part_win_x[l], part_win_y[l]);
      }
      fprintf(fplog, "%d}", labeln);
      if ((l==0) && (((int) act[l][1])==act[l][1])) {
        for (i=1; i < n[l]+1; i++) {
	  fprintf(fplog, "%1.0f,", act[l][i]);
        }
      } else {
        for (i=1; i < n[l]+1; i++) {
	  fprintf(fplog, "%6.4f,", act[l][i]);
        }
      }
      if (DIM_SOM[l]>0) {
	fprintf(fplog, "Reli %3d, Dist %5.2f",
	  part_winner_reliability[l], part_winner_dis[l]);
      }
    }
  }
  if (show[LAST_LAYER]>0) {
    fprintf(fplog, "\nTT: ");
    for (i=1; i < n[LAST_LAYER]+1; i++) {
      fprintf(fplog, "%6.4f,", target[i]);
    }
  }
}

void fatal_error(void)
{
  fprintf(fplog, "\n\nEXIT: fatal error");
  fclose(fplog);
  exit(0);
}

void reset_average_act(void)
{
  int l, i;
  for (l=1; l<LAST_LAYER; l++) {
    for (i=1; i<=n[l]; i++) { average_act[l][i]=0.0; }
  }
}

void update_average_act(void)
{
  int l, i;
  for (l=1; l<LAST_LAYER; l++) {
    for (i=1; i<=n[l]; i++) {
        average_act[l][i]+=act[l][i];
    }
  }
}

void ress_average_actc(long int number)
{
  int l, i;

  for (l=1; l<LAST_LAYER; l++) {
    for (i=1; i<=n[l]; i++) {
      average_actc[l][i]=(average_act[l][i]/number);
    }
  }
}

void reset_ss(void)
{
  int l, i, j;

  for (l=1; l<LAST_LAYER; l++) {
    for (i=1; i<=n[l]; i++) {
      ss_act[l][i]=0.0;
      if (CORR) {
	for (j=1; j<=n[l]; j++) {
          ss_xy[l][i][j]=0.0;
        }
      }
    }
  }
}

void clear_prune_information(void)
{
  int l;
  for (l=1; l<LAST_LAYER; l++) { n_pruned_units[l]=0; }
}

void prune_if_possible(void)
{
  int l,i;

  for (l=1; l<LAST_LAYER; l++) {
    if (n[l]>1) {
      for (i=1; i<=n[l]; i++) {
	if (std_act[l][i]<=PRUNE_THRESHOLD) {
           n_pruned_units[l]++;
	   printf("-%d ", n_pruned_units[l]);
           fprintf(fplog, "%d>%d ", l, i);
           prune_weights(l,i);
           nothing_done=0;
        }
      }
    }
  }
}

void prune_weights(int l, int w)
{
  int i, j, x, y;
  for (i=0; i<=n[l-1]; i++) {
    for (j=w; j<n[l]; j++) {
      weight[l-1][i][j]=weight[l-1][i][j+1];
      del_old[l-1][i][j]=del_old[l-1][i][j+1];
    }
  }
  for (i=1; i<=n[l+1]; i++) {
    weight[l][0][i]=weight[l][0][i]+
      (average_actc[l][w]*weight[l][w][i]);
  }
  for (i=w; i<n[l]; i++) {
    for (j=1; j<=n[l+1]; j++) {
      weight[l][i][j]=weight[l][i+1][j];
      del_old[l][i][j]=del_old[l][i+1][j];
    }
  }
  for (x=1; x<=DIM_SOM[l]; x++) {
    for (y=1; y<=DIM_SOM[l]; y++) {
      for (i=w; i<n[l]; i++) {
	som_network_vector[l][x][y][i]=som_network_vector[l][x][y][i+1];
      }
    }
  }
  for (i=w; i<n[l]; i++) {
    average_act[l][i]=average_act[l][i+1];
    average_actc[l][i]=average_actc[l][i+1];
    std_act[l][i]=std_act[l][i+1];
    ss_act[l][i]=ss_act[l][i+1];
  }
  n[l]=n[l]-1;
}

void update_ss(void)
{
  int l, i, j;
  for (l=1; l<LAST_LAYER; l++) {
    for (i=1; i<=n[l]; i++) {
      ss_act[l][i]+=((act[l][i]-average_actc[l][i]) *
	             (act[l][i]-average_actc[l][i]));
      if (CORR) {
	for (j=1; j<=n[l]; j++) {
	  ss_xy[l][i][j]+=((act[l][i]-average_actc[l][i]) *
		 	   (act[l][j]-average_actc[l][j]));
	}
      }
    }
  }
}

void ss_to_std(long int teller)
{
  int l, j;
  for (l=1; l<LAST_LAYER; l++) {
    for (j=1; j<=n[l]; j++) {
      std_act[l][j]=(float) sqrt( (double) (ss_act[l][j]/teller));
    }
  }
}

void record_corr(void)
{
  int l, i, j;
  float ress;

  for (l=1; l<LAST_LAYER; l++) {
    fprintf(fplog, "\n\nHidden layer %d:\n      ", l);
    fprintf(fplog, "\n\nSTD:  ");
    for (j=1; j<=n[l]; j++) {
      fprintf(fplog, "%5.3f ", std_act[l][j]);
    }
    fprintf(fplog, "\nAVG   ");
    for (j=1; j<=n[l]; j++) {
      fprintf(fplog, "%5.3f ", average_actc[l][j]);
    }
    fprintf(fplog, "\n      ");
    for (j=1; j<=n[l]; j++) {
      fprintf(fplog, "%5d ", j);
    }
    if (CORR) {
      for (i=1; i<=n[l]; i++) {
        fprintf(fplog, "\n%5d ", i);
        for (j=1; j<=n[l]; j++) {
          if ((ss_act[l][i]==0) || (ss_act[l][j]==0)) {
            ress=0.0;
          } else {
            ress=ss_xy[l][i][j]/sqrt((double) (ss_act[l][i]*ss_act[l][j]));
          }
        fprintf(fplog, "%5.2f ", ress);
        }
      }
    }
  }
}

void histogram_teller(float weight)
{
  if (weight<-9.5) {
    column[0]++;
  } else {
    if (weight>9.5) {
      column[20]++;
    } else {
      column[((int) weight) + 10]++;
    }
  }
}

void record_histogram(void)
{
  int i, j, k, n_weights=0;

  for (k=0; k<=20; k++) {
    column[k]=0;
  }

  for (i=0; i<LAST_LAYER; i++) {
    for (j=0; j<n[i]+1; j++) {
      for (k=1; k<n[i+1]+1; k++) {
        histogram_teller(weight[i][j][k]);
        n_weights++;
      }
    }
  }

  for (k=0; k<=20; k++) {
    fprintf(fplog, "\n%3d ", k-10);
    if (DEBUG) printf("\n%3d ", k-10);
    for (i=1; i<=(int) (100.0*column[k]/n_weights); i++) {
      fprintf(fplog, "x");
      if (DEBUG) printf("x");
    }
  }
}
