import { DialogDescription } from "@radix-ui/react-dialog";
import { useForm } from "@tanstack/react-form";
import { Plus, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { z } from "zod";
import {
  useAddProviderModel,
  useCheckModelAvailability,
  useDeleteProviderModel,
  useGetModelProviderDetail,
  useLoginModelProviderOAuth,
  useLogoutModelProviderOAuth,
  useSetDefaultProvider,
  useSetDefaultProviderModel,
  useUpdateProviderConfig,
} from "@/api/setting";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Field,
  FieldError,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

const configSchema = z.object({
  base_url: z.string(),
});

const addModelSchema = z.object({
  model_id: z.string().min(1, "Model ID is required"),
  model_name: z.string().min(1, "Model name is required"),
});

type ModelDetailProps = {
  provider: string;
};

export function ModelDetail({ provider }: ModelDetailProps) {
  const { t } = useTranslation();

  const { data: providerDetail, isLoading: detailLoading } =
    useGetModelProviderDetail(provider);
  const { mutate: updateConfig, isPending: updatingConfig } =
    useUpdateProviderConfig();
  const { mutateAsync: loginOAuth, isPending: loggingIn } =
    useLoginModelProviderOAuth();
  const { mutateAsync: logoutOAuth, isPending: loggingOut } =
    useLogoutModelProviderOAuth();
  const { mutate: addModel, isPending: addingModel } = useAddProviderModel();
  const { mutate: deleteModel, isPending: deletingModel } =
    useDeleteProviderModel();
  const { mutate: setDefaultModel, isPending: settingDefaultModel } =
    useSetDefaultProviderModel();
  const { mutate: setDefaultProvider, isPending: settingDefaultProvider } =
    useSetDefaultProvider();
  const {
    data: checkResult,
    mutateAsync: checkAvailability,
    isPending: checkingAvailability,
    reset: resetCheckResult,
  } = useCheckModelAvailability();

  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);

  const configForm = useForm({
    defaultValues: {
      base_url: "",
    },
    validators: {
      onSubmit: configSchema,
    },
    onSubmit: async ({ value }) => {
      if (!provider) return;
      updateConfig({
        provider,
        base_url: value.base_url,
      });
    },
  });

  useEffect(() => {
    if (providerDetail) {
      configForm.setFieldValue("base_url", providerDetail.base_url || "");
    }
  }, [providerDetail, configForm.setFieldValue]);

  useEffect(() => {
    if (provider) {
      resetCheckResult();
    }
  }, [provider, resetCheckResult]);

  const addModelForm = useForm({
    defaultValues: {
      model_id: "",
      model_name: "",
    },
    validators: {
      onSubmit: addModelSchema,
    },
    onSubmit: async ({ value }) => {
      if (!provider) return;
      addModel({
        provider,
        model_id: value.model_id,
        model_name: value.model_name,
      });
      addModelForm.reset();
      setIsAddDialogOpen(false);
    },
  });

  const handleSetDefaultModel = (modelId: string) => {
    if (!provider) return;
    setDefaultModel({ provider, model_id: modelId });
  };

  const handleDeleteModel = (modelId: string) => {
    if (!provider) return;
    deleteModel({ provider, model_id: modelId });
  };

  const isBusy =
    updatingConfig ||
    addingModel ||
    deletingModel ||
    settingDefaultModel ||
    settingDefaultProvider ||
    checkingAvailability ||
    loggingIn ||
    loggingOut;

  if (detailLoading) {
    return (
      <div className="text-muted-foreground text-sm">
        {t("settings.models.loading")}
      </div>
    );
  }

  if (!providerDetail) {
    return null;
  }

  return (
    <div className="scroll-container flex flex-1 flex-col px-8">
      <div className="mb-4 flex items-center justify-between">
        <p className="font-semibold text-foreground text-lg">
          {t(`strategy.providers.${provider}`) || provider}
        </p>
        <div className="flex items-center gap-2">
          <p className="font-semibold text-base text-muted-foreground">
            {t("settings.models.defaultProvider")}
          </p>
          <Switch
            checked={providerDetail.is_default}
            disabled={isBusy}
            onCheckedChange={() => setDefaultProvider({ provider })}
          />
        </div>
      </div>

      <form>
        <div className="flex flex-col gap-6">
          <FieldGroup>
            <Field className="text-foreground">
              <FieldLabel className="font-medium text-base">
                ChatGPT OAuth
              </FieldLabel>
              <div className="flex flex-col gap-3 rounded-lg border border-border bg-card px-4 py-3">
                <div className="text-sm">
                  {providerDetail.oauth_authenticated ? (
                    <span className="text-green-600">
                      Connected
                      {providerDetail.oauth_expires_at
                        ? `, expires ${new Date(providerDetail.oauth_expires_at).toLocaleString()}`
                        : ""}
                    </span>
                  ) : (
                    <span className="text-muted-foreground">Not connected</span>
                  )}
                </div>
                <div className="flex flex-wrap gap-2">
                  {!providerDetail.oauth_authenticated ? (
                    <Button
                      type="button"
                      disabled={isBusy}
                      onClick={async () => {
                        await loginOAuth({ provider });
                      }}
                    >
                      {loggingIn ? "Connecting..." : "Connect ChatGPT"}
                    </Button>
                  ) : (
                    <Button
                      type="button"
                      variant="outline"
                      disabled={isBusy}
                      onClick={async () => {
                        await logoutOAuth({ provider });
                      }}
                    >
                      {loggingOut ? "Disconnecting..." : "Disconnect"}
                    </Button>
                  )}
                  <Button
                    type="button"
                    variant="outline"
                    disabled={isBusy}
                    onClick={async () => {
                      await checkAvailability({
                        provider,
                        model_id: providerDetail.default_model_id || undefined,
                      });
                    }}
                  >
                    {checkingAvailability ? "Checking..." : "Check Availability"}
                  </Button>
                </div>
                {checkResult?.data && (
                  <div className="text-sm">
                    {checkResult.data.ok ? (
                      <span className="text-green-600">
                        Reachable
                        {checkResult.data.status
                          ? ` (${checkResult.data.status})`
                          : ""}
                      </span>
                    ) : (
                      <span className="text-red-600">
                        Unavailable
                        {checkResult.data.status
                          ? ` (${checkResult.data.status})`
                          : ""}
                        {checkResult.data.error
                          ? `: ${checkResult.data.error}`
                          : ""}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </Field>

            {/* API Host section */}
            <configForm.Field name="base_url">
              {(field) => (
                <Field className="text-foreground">
                  <FieldLabel className="font-medium text-base">
                    {t("settings.models.apiHost")}
                  </FieldLabel>
                  <Input
                    placeholder={providerDetail.base_url}
                    value={field.state.value}
                    onChange={(e) => field.handleChange(e.target.value)}
                    onBlur={() => configForm.handleSubmit()}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        e.currentTarget.blur();
                      }
                    }}
                  />
                  <FieldError errors={field.state.meta.errors} />
                </Field>
              )}
            </configForm.Field>
          </FieldGroup>

          {/* Models section */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div className="font-medium text-base text-foreground">
                {t("settings.models.models")}
              </div>
              <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
                <DialogTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 border-border px-2.5 font-semibold text-muted-foreground text-sm"
                    disabled={isBusy}
                  >
                    <Plus className="size-4" />
                    {t("settings.models.add")}
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      addModelForm.handleSubmit();
                    }}
                  >
                    <DialogHeader>
                      <DialogTitle>{t("settings.models.addModel")}</DialogTitle>
                      <DialogDescription />
                    </DialogHeader>
                    <div className="flex flex-col gap-4 py-4">
                      <FieldGroup className="gap-4">
                        <addModelForm.Field name="model_id">
                          {(field) => (
                            <Field>
                              <FieldLabel className="font-medium text-sm">
                                {t("settings.models.modelId")}
                              </FieldLabel>
                              <Input
                                placeholder={t("settings.models.enterModelId")}
                                value={field.state.value}
                                onChange={(e) =>
                                  field.handleChange(e.target.value)
                                }
                                onBlur={field.handleBlur}
                              />
                              <FieldError errors={field.state.meta.errors} />
                            </Field>
                          )}
                        </addModelForm.Field>

                        <addModelForm.Field name="model_name">
                          {(field) => (
                            <Field>
                              <FieldLabel className="font-medium text-sm">
                                {t("settings.models.modelName")}
                              </FieldLabel>
                              <Input
                                placeholder={t(
                                  "settings.models.enterModelName",
                                )}
                                value={field.state.value}
                                onChange={(e) =>
                                  field.handleChange(e.target.value)
                                }
                                onBlur={field.handleBlur}
                              />
                              <FieldError errors={field.state.meta.errors} />
                            </Field>
                          )}
                        </addModelForm.Field>
                      </FieldGroup>
                    </div>
                    <DialogFooter>
                      <Button
                        className="flex-1"
                        type="button"
                        variant="outline"
                        onClick={() => {
                          addModelForm.reset();
                          setIsAddDialogOpen(false);
                        }}
                      >
                        {t("settings.models.cancel")}
                      </Button>
                      <Button
                        className="flex-1"
                        type="submit"
                        disabled={isBusy || !addModelForm.state.canSubmit}
                      >
                        {t("settings.models.confirm")}
                      </Button>
                    </DialogFooter>
                  </form>
                </DialogContent>
              </Dialog>
            </div>

            {providerDetail.models.length === 0 ? (
              <div className="rounded-lg border border-border border-dashed p-4 text-muted-foreground text-sm">
                {t("settings.models.noModels")}
              </div>
            ) : (
              <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3">
                {providerDetail.models.map((m) => (
                  <div
                    key={m.model_id}
                    className="flex items-center justify-between"
                  >
                    <span className="font-normal text-foreground text-sm">
                      {m.model_name}
                    </span>

                    <div className="flex items-center gap-3">
                      <Switch
                        className="cursor-pointer"
                        checked={m.model_id === providerDetail.default_model_id}
                        disabled={isBusy}
                        onCheckedChange={() =>
                          handleSetDefaultModel(m.model_id)
                        }
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        disabled={isBusy}
                        onClick={() => handleDeleteModel(m.model_id)}
                      >
                        <Trash2 className="size-5" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}
